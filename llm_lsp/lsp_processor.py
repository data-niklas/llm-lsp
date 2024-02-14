from transformers import LogitsProcessor, PreTrainedTokenizer
from torch import LongTensor, FloatTensor, full_like, full
import torch
from pygls.lsp.client import BaseLanguageClient
from logzero import logger
from lsprotocol.types import *
import asyncio
from os import path
import os
import uuid
from functools import lru_cache
import json
import re

# TODO: add special token
# TODO: use special token as interrupt to provide more information to pipeline
from llm_lsp.constants import DEPRECATION_INTERRUPT_TOKEN, SIGNATURE_INTERRUPT_TOKEN
from llm_lsp.deprecation_messages import is_deprecated


def custom_completion_item_hash(self):
    return hash((self.label, self.kind))


CompletionItem.__hash__ = custom_completion_item_hash


class CompletionItemRanker:
    def __init__(self, pipeline, tokenizer):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.configuration = {
            # "num_beam_groups": 0,
            "num_beams": 2,
            # "diversity_penalty": 1.0,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "num_return_sequences": 1,
            "return_full_text": False,
            "temperature": 0.7,
            "max_new_tokens": 2048,
        }
        self.cache = {}
        self.categorize_prompt_template = "[INST] Categorize the goal and usage of the provided code piece. In addition to the code, you will receive the documentation and further details. Return the result as a JSON array of strings. Wrap strings in \". Do not return further comments, ONLY return the JSON array. Here is an example:\nCODE: 'to_dict(self)'\nDOCUMENTATION: 'Convert an instance of self to a Python dictionary'\nDETAILS: ''\nRESULT: [\"convert\", \"python\"]\n[/INST]\n"
        self.rank_prompt_template = "[INST] You will be provided with code, a code snippet and the tags associated with the code snippet. Rank the relevance of the code piece to the previous code based on its tags. '1' is very relevant, whilst '0' is unrelevant. Return only the score as a literal. Do not return further comments.\n[/INST]\n"
        # TODO: provide example in rank_prompt_template
        # TODO: mention scores between 0 and 1

    def summarize_completions(self, completions: List[CompletionItem]) -> List[str]:
        """Returns a list of tags for the completion"""

        def create_prompt():
            for completion in completions:
                yield self.categorize_prompt_template + f"CODE: '{completion.label}'\nDOCUMENTATION: '{completion.documentation.value}'\nDETAILS: '{completion.detail}'"

        def get_tags(sequences, completion):
            result_text = sequences[0]["generated_text"].strip()
            if result_text.startswith("RESULT:"):
                result_text = result_text[7:].strip()
            if result_text.startswith("JSON ARRAY:"):
                result_text = result_text[10:].strip()
            try:
                tags = list(set(json.loads(result_text)))
                # logger.debug("TAGS: [" + ", ".join(tags) + "]")
                return tags
            except json.JSONDecodeError:
                logger.error(result_text)
                return [completion.insert_text]

        return [
            get_tags(result, completion)
            for result, completion in zip(
                self.pipeline(
                    create_prompt(),
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **self.configuration,
                    pad_token_id=self.tokenizer.eos_token_id,
                ),
                completions,
            )
        ]

    def rank_completions(self, code, completions):
        """Returns a list of tuples of completions with their score"""
        completion_tags = self.summarize_completions(completions)

        def create_prompt():
            for tags, completion in zip(completion_tags, completions):
                snippet = completion.label
                tags = ", ".join(tags)
                yield self.rank_prompt_template + f"CODE:\n```\n{code}\n```\nSNIPPET: '{snippet}'\nTAGS: '{tags}'"

        def get_score(sequences, completion):
            result_text = sequences[0]["generated_text"].strip()
            if result_text.startswith("RESULT:"):
                result_text = result_text[7:].strip()
            if result_text.startswith("SCORE:"):
                result_text = result_text[6:].strip()
            try:
                return float(result_text)
            except ValueError:
                logger.error("Could not parse: '" + result_text + "'")
                return 0

        return [
            get_score(result, completion)
            for result, completion in zip(
                self.pipeline(
                    create_prompt(),
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **self.configuration,
                    pad_token_id=self.tokenizer.eos_token_id,
                ),
                completions,
            )
        ]


class LspLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, lsp_client, prompt_len, file, code, pipeline):
        tokenizer.add_special_tokens({})
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.lsp_client: BaseLanguageClient = lsp_client
        self.prompt_len = prompt_len
        self.file = file
        self.code = code
        self.completion_ranker = CompletionItemRanker(pipeline, tokenizer)

    def ids_to_text(self, input_ids: LongTensor) -> str:
        return self.tokenizer.decode(input_ids)

    def current_code(self, input_ids: LongTensor) -> str:
        """The complete generated code at this point. This includes the starting code and the generated code."""
        generated_code_with_prompt = self.ids_to_text(input_ids)
        generated_code = generated_code_with_prompt[self.prompt_len :]
        return self.code + "\n" + generated_code

    def completion_text(self, completion) -> str:
        return completion.insert_text or completion.label

    def filter_completions_by_kind(self, completions):
        return [
            completion
            for completion in completions
            if completion.kind
            in [
                CompletionItemKind.Method,
                CompletionItemKind.Field,
                CompletionItemKind.Class,
                CompletionItemKind.Function,
                CompletionItemKind.Property,
                CompletionItemKind.Variable
            ]
        ]

    def filter_builtin_completions(self, completions):
        return [
            completion for completion in completions if completion.detail != "builtins"
        ]

    def filter_completions_by_case(self, code: str, completions):
        """Remove already completed completions"""
        # rsplit
        trigger_phrase = re.search(r'[A-Za-z_]*$', code).group(0)
        trigger_phrase_lower = trigger_phrase.lower()

        return [
            completion
            for completion in completions
            if not completion.insert_text.lower().startswith(trigger_phrase_lower)
            or completion.insert_text.startswith(trigger_phrase)
        ]

    def filter_misc(self, completions):
        return [
            completion
            for completion in completions
            if not self.completion_text(completion).startswith("__")
            or self.completion_text(completion) in ["__getitem__", "__setitem__"]
        ]

    def is_deprecated(self, completion) -> bool:
        return is_deprecated(completion.detail + "." + completion.insert_text)

    def split_deprecated_completions(self, completions):
        non_deprecated = []
        deprecated = []
        for completion in completions:
            if self.is_deprecated(completion):
                deprecated.append(completion)
            else:
                non_deprecated.append(completion)
        return non_deprecated, deprecated

    def tokenize_completion(self, code_token_len, code: str, completion):
        completion_text = self.completion_text(completion)
        return self.tokenizer(
            code + completion_text, add_special_tokens=False
        ).input_ids[code_token_len:]

    def tokenize_completions(self, code: str, completions):
        code_token_len = len(self.tokenizer(code, add_special_tokens=False).input_ids)
        return [
            self.tokenize_completion(code_token_len, code, completion)
            for completion in completions
        ]

    def index_of_last_matching_token(self, input_ids, tokens) -> int:
        # Algorithm is not perfect, as .index returns the first match, but the token could occur twice
        try:
            input_ids_index = -2
            last_input_id_index = tokens.index(input_ids[-1])
            for i in reversed(range(last_input_id_index)):
                if input_ids[input_ids_index] != tokens[i]:
                    return -1
                input_ids_index -= 1
            text = self.tokenizer.decode(input_ids[-min(len(input_ids), 6) :])
            return last_input_id_index
        except ValueError:
            return -1
        except IndexError:
            return -1

    def get_unique_first_tokens(self, indexes_after_overlap, tokens):
        result_tokens = []
        for index, tokens in zip(indexes_after_overlap, tokens):
            if index >= len(tokens):
                # For now just assume everything is a function
                paren_open_id = self.tokenizer(
                    self.tokenizer.decode(tokens) + "("
                ).input_ids[-1]
                result_tokens.append(paren_open_id)
                continue
            result_tokens.append(tokens[index])
        return list(set(result_tokens))

    def count_full_match(self, indexes_after_overlap, tokens):
        return sum(
            [
                1 if index >= len(tokens) else 0
                for index, tokens in zip(indexes_after_overlap, tokens)
            ]
        )

    def index_of_last_matching_tokens(self, input_ids, tokens):
        return [
            self.index_of_last_matching_token(input_ids, completion_tokens) + 1
            for completion_tokens in tokens
        ]

    def ensure_deprecated_below_non_deprecated(
        self, scores, non_deprecated, deprecated
    ):
        minimum = 10000.0
        for token in non_deprecated:
            minimum = min(minimum, scores[token].item())
        for token in deprecated:
            scores[token] = min(minimum - 7, scores[token].item())
        return scores

    def downrank_completed_items(
        self,
        scores,
        non_deprecated_index_after_last_overlapping_token,
        non_deprecated_tokens,
    ):
        non_deprecated_unique_first_tokens = self.get_unique_first_tokens(
            non_deprecated_index_after_last_overlapping_token, non_deprecated_tokens
        )
        for token in non_deprecated_unique_first_tokens:
            scores[token] -= 14
        return scores

    def apply_constant_adjustments(
        self, scores, non_deprecated_unique_first_tokens, deprecated_unique_first_tokens
    ):
        for non_deprecated_token in non_deprecated_unique_first_tokens:
            scores[non_deprecated_token] += 7.0
        for deprecated_token in deprecated_unique_first_tokens:
            scores[deprecated_token] -= 7.0
        return scores

    def filter_tokens_by_overlaps(
        self, index_after_last_overlapping_token, tokens, completions
    ):
        # If there is at least one item with overlap, then the model is walking towards the overlap and all other items without overlap are irrelevant
        if len(index_after_last_overlapping_token) == 0:
            return index_after_last_overlapping_token, tokens, completions
        largest_index = max(index_after_last_overlapping_token)
        filtered_items = [
            (index, tokens, completion)
            for index, tokens, completion in zip(
                index_after_last_overlapping_token, tokens, completions
            )
            if index == largest_index
        ]
        index_after_last_overlapping_token, tokens, completions = zip(*filtered_items)
        return index_after_last_overlapping_token, tokens, completions

    def uprank_divider_after_completion(self, scores, input_ids):
        text = self.tokenizer.decode(input_ids[-1:])
        open_token = self.tokenizer(text + "(").input_ids[-1]
        scores[open_token] += 14
        return scores

    def check_deprecation_documentation_included(
        self, current_code: str, deprecated_completions
    ):
        if len(deprecated_completions) == 0:
            return True
        lines = current_code.splitlines()[:-1]
        lines.reverse()
        for line in lines:
            line = line.strip()
            if not line.startswith("# "):
                return False
            elif line.startswith("# Deprecation note: "):
                return True
        return False

    def check_signature_documentation_included(self, current_code: str, signature_help):
        if signature_help is None or len(signature_help.signatures) == 0:
            return True
        first_signature = signature_help.signatures[
            signature_help.active_signature
        ].label
        lines = current_code.splitlines()[:-1]
        lines.reverse()
        for line in lines:
            line = line.strip()
            if not line.startswith("# "):
                return False
            elif line.startswith("# Signature note: "):
                return True
        return False

    def should_complete(self, code):
        symbols = [")", "\n", " ", "\t", "]", "}"]
        for symbol in symbols:
            if code.endswith(symbol):
                return False
        return True

    def interrupt(self, scores, data, i):
        input_id = self.tokenizer.convert_tokens_to_ids(i)
        scores[input_id] = 100
        self.interrupt_data = data
        return scores

    def scores_for_batch(
        self, input_ids: LongTensor, scores: FloatTensor
    ) -> FloatTensor:
        """Returns a 1d FloatTensor for a single batch"""
        current_code = self.current_code(input_ids)
        # logger.debug("CODE")
        # logger.debug(current_code)
        with LspCodeFile(self.file, current_code, self.lsp_client) as lsp_code_file:
            if self.should_complete(current_code):
                resolved_completions = lsp_code_file.ask_completions()
            else:
                resolved_completions = []
            signature_help = lsp_code_file.ask_signature()
            filtered_completions = self.filter_misc(
                self.filter_completions_by_case(
                    current_code,
                    self.filter_completions_by_kind(
                        self.filter_builtin_completions(resolved_completions)
                    ),
                )
            )
            (
                non_deprecated_completions,
                deprecated_completions,
            ) = self.split_deprecated_completions(filtered_completions)
            if not self.check_deprecation_documentation_included(
                current_code, deprecated_completions
            ):
                scores = self.interrupt(
                    scores, deprecated_completions, DEPRECATION_INTERRUPT_TOKEN
                )
                logger.debug("Interrupting code at: ")
                logger.debug(current_code)
                return scores
            if not self.check_signature_documentation_included(
                current_code, signature_help
            ):
                scores = self.interrupt(
                    scores, signature_help, SIGNATURE_INTERRUPT_TOKEN
                )
                logger.debug("Interrupting code at: ")
                logger.debug(current_code)
                return scores
            # Downranking etc needed, else it won't choose the alternative
            non_deprecated_tokens, deprecated_tokens = self.tokenize_completions(
                current_code, non_deprecated_completions
            ), self.tokenize_completions(current_code, deprecated_completions)
            (
                non_deprecated_index_after_last_overlapping_token,
                deprecated_index_after_last_overlapping_token,
            ) = self.index_of_last_matching_tokens(
                input_ids, non_deprecated_tokens
            ), self.index_of_last_matching_tokens(
                input_ids, deprecated_tokens
            )
            # Only filter non deprecated, as deprecated should be downranked nonetheless
            (
                non_deprecated_index_after_last_overlapping_token,
                non_deprecated_tokens,
                non_deprecated_completions,
            ) = self.filter_tokens_by_overlaps(
                non_deprecated_index_after_last_overlapping_token,
                non_deprecated_tokens,
                non_deprecated_completions,
            )
            (
                non_deprecated_unique_first_tokens,
                deprecated_unique_first_tokens,
            ) = self.get_unique_first_tokens(
                non_deprecated_index_after_last_overlapping_token, non_deprecated_tokens
            ), self.get_unique_first_tokens(
                deprecated_index_after_last_overlapping_token, deprecated_tokens
            )
            # if len(non_deprecated_completions) > 0:
            #    c_scores = self.completion_ranker.rank_completions(current_code, non_deprecated_completions)
            #    logger.debug(list(zip(c_scores, [c.label for c in non_deprecated_completions])))
            scores = self.apply_constant_adjustments(
                scores,
                non_deprecated_unique_first_tokens,
                deprecated_unique_first_tokens,
            )
            scores = self.ensure_deprecated_below_non_deprecated(
                scores,
                non_deprecated_unique_first_tokens,
                deprecated_unique_first_tokens,
            )
        return scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        """Returns a 2d FloatTensor which has scores for every batch"""
        if os.getenv("DISABLE") is not None:
            return scores
        for i in range(input_ids.shape[0]):
            scores[i] = self.scores_for_batch(input_ids[i], scores[i])
        return scores


class LspCodeFile:
    def __init__(self, file, code, lsp_client):
        self.path = file + "_" + str(uuid.uuid1())
        self.uri = "file://" + path.abspath(self.path)
        self.text_document_item = TextDocumentItem(
            uri=self.uri,
            language_id="python",
            version=1,
            text=code,
        )
        self.lsp_client = lsp_client
        self.code = code

    def __enter__(self):
        with open(self.path, "w") as f:
            f.write(self.code)
        self.lsp_client.text_document_did_open(
            DidOpenTextDocumentParams(text_document=self.text_document_item)
        )
        return self

    def __exit__(self, _a, _b, _c):
        self.lsp_client.text_document_did_close(
            DidCloseTextDocumentParams(TextDocumentIdentifier(uri=self.uri))
        )
        os.remove(self.path)

    def ask_completions(self):
        char, line = self.char_line_of_code(self.code)
        completion_awaitable = self.lsp_client.text_document_completion_async(
            CompletionParams(
                text_document=self.text_document_item,
                position=Position(character=char, line=line),
                context=CompletionContext(trigger_kind=CompletionTriggerKind.Invoked),
            )
        )
        completions = asyncio.get_event_loop().run_until_complete(completion_awaitable)
        if isinstance(completions, CompletionList):
            completions = completions.items
        resolved_completions = [
            asyncio.get_event_loop().run_until_complete(
                self.lsp_client.completion_item_resolve_async(completion)
            )
            for completion in completions
        ]
        return resolved_completions

    def ask_signature(self):
        char, line = self.char_line_of_code(self.code)
        signature_awaitable = self.lsp_client.text_document_signature_help_async(
            SignatureHelpParams(
                text_document=self.text_document_item,
                position=Position(character=char, line=line),
                context=None,
            )
        )
        return asyncio.get_event_loop().run_until_complete(signature_awaitable)

    def char_line_of_code(self, code):
        lines = code.splitlines()
        last_line_index = len(lines) - 1
        last_line = lines[last_line_index]
        last_char_index = len(last_line)
        return max(last_char_index, 0), max(last_line_index, 0)
