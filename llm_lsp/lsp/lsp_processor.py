import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List

import torch
from lsprotocol.types import *  # noqa: F403
from pygls.lsp.client import BaseLanguageClient
from torch import FloatTensor, LongTensor
from transformers import LogitsProcessor

from llm_lsp.config import LspGenerationConfig
from llm_lsp.interrupts import Interrupt
from llm_lsp.interrupts.completion import COMPLETION_COMMENT_TYPE
from llm_lsp.interrupts.deprecation import DEPRECATION_COMMENT_TYPE, is_deprecated, are_deprecated
from llm_lsp.interrupts.signature import SIGNATURE_COMMENT_TYPE
from llm_lsp.lsp.file import LspCodeFile

# Was inf but caused issues
INTERRUPT_LOGITS_SCORE = 1000.0
RANK_DELTA = 7.0


@dataclass
class InterruptGeneration(BaseException):
    interrupt_type_name: str
    context: Any


def custom_completion_item_hash(self):
    return hash((self.label, self.kind))


CompletionItem.__hash__ = custom_completion_item_hash


def eq_completions_items(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.insert_text != y.insert_text:
            return False
    return True


def in_completion_items(a, b):
    a = set([x.label for x in a])
    b = set([x.label for x in b])
    return a <= b


def eq_signature_help(a, b):
    a = a.signatures
    b = b.signatures
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.label != y.label:
            return False
    return True


class LspLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        lsp_clients,
        prompt_states,
        file_names,
        interrupt_token_id,
        expand_size,
        beam_tracker,
        config: LspGenerationConfig,
    ):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.lsp_clients: BaseLanguageClient = lsp_clients
        self.prompt_states = prompt_states
        self.file_names = file_names
        self.signature_cache = {}
        self.expand_size = expand_size
        self.interrupt_token_id = interrupt_token_id
        self.interrupt = None
        self.beam_tracker = beam_tracker
        self.config = config

    def ids_to_text(self, input_ids: LongTensor) -> str:
        # remove padding
        token_id = self.tokenizer.pad_token_id
        index = (input_ids != token_id).nonzero()[0].item()
        input_ids = input_ids[index:]
        return self.tokenizer.decode(input_ids, skip_special_tokens=False)

    def current_code(self, i, input_ids: LongTensor) -> str:
        """The complete generated code at this point. This includes the starting code and the generated code."""
        generated_code_with_prompt = self.ids_to_text(input_ids)
        prompt_state_index = i
        if self.beam_tracker.is_beam_search():
            prompt_state_index = self.beam_tracker.get_final_beam_indices()[i]
        code = self.prompt_states[prompt_state_index].get_whole_code(
            generated_code_with_prompt
        )
        return code

    def completion_text(self, completion) -> str:
        return completion.insert_text or completion.label

    def is_builtin_completion(self, completion):
        return completion.detail == "builtins"

    def filter_builtin_completions(self, completions):
        return [
            completion for completion in completions if not self.is_builtin_completion(completion)
        ]

    def filter_signatures_without_parameters(self, completions):
        return [
            completion
            for completion in completions
            if completion.parameters is not None
        ]

    def filter_completions_by_case(self, trigger_phrase: str, completions):
        """Remove already completed completions"""
        # rsplit
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
            if not self.completion_text(completion).startswith("_")
            or self.completion_text(completion) in ["__getitem__", "__setitem__"]
        ]

    def is_deprecated(self, completion) -> bool:
        return is_deprecated(completion.detail + "." + completion.insert_text)

    def are_deprecated(self, completions) -> bool:
        return are_deprecated([completion.detail + "." + completion.insert_text for completion in completions])

    def split_deprecated_completions(self, completions):
        non_deprecated = []
        deprecated = []
        if len(completions) == 0:
            return non_deprecated, deprecated
        for completion, is_completion_deprecated in zip(completions, self.are_deprecated(completions)):
            if is_completion_deprecated:
                deprecated.append(completion)
            else:
                non_deprecated.append(completion)
        return non_deprecated, deprecated

    @lru_cache
    def tokenize_overlap(self, code_token_len, code: str, overlap):
        return self.tokenizer(code + overlap, add_special_tokens=False).input_ids[
            code_token_len:
        ]

    def overlap_unique_first_tokens(self, code: str, overlaps):
        # Assume that by concatenating the chosen tokens do not change
        code_token_len = len(self.tokenizer(code, add_special_tokens=False).input_ids)
        return list(
            set(
                [
                    self.tokenize_overlap(code_token_len, code, overlap)[0]
                    for overlap in overlaps
                    if len(self.tokenize_overlap(code_token_len, code, overlap)) > 0
                ]
            )
        )

    def count_full_match(self, indexes_after_overlap, tokens):
        return sum(
            [
                1 if index >= len(tokens) else 0
                for index, tokens in zip(indexes_after_overlap, tokens)
            ]
        )

    def ensure_deprecated_below_non_deprecated(
        self, scores, non_deprecated, deprecated
    ):
        minimum = 50.0
        for token in non_deprecated:
            minimum = min(minimum, scores[token].item())
        for token in deprecated:
            scores[token] = min(minimum - RANK_DELTA, scores[token].item())
        return scores

    def mask_other_tokens(self, scores, non_deprecated):
        if len(non_deprecated) == 0:
            return scores
        return torch.where(
            torch.tensor(
                [True if i in non_deprecated else False for i in range(scores.shape[0])]
            ),
            scores,
            float("-inf") * torch.ones(scores.shape[0]),
        )

    def apply_constant_adjustments(
        self, scores, non_deprecated_unique_first_tokens, deprecated_unique_first_tokens
    ):
        for non_deprecated_token in non_deprecated_unique_first_tokens:
            scores[non_deprecated_token] += RANK_DELTA
        for deprecated_token in deprecated_unique_first_tokens:
            scores[deprecated_token] -= RANK_DELTA
        return scores

    def filter_uri(self, completions, uri):
        uri_name = uri.split("/")[-1]
        return [c for c in completions if not c.detail.startswith(uri_name)]

    def filter_completions_by_postfix(self, trigger_phrase: str, completions):
        return [
            completion
            for completion in completions
            if completion.kind
            in [
                CompletionItemKind.Method,
                CompletionItemKind.Field,
                CompletionItemKind.Function,
                CompletionItemKind.Property,
                CompletionItemKind.TypeParameter,
            ]
            or (
                completion.insert_text.startswith(trigger_phrase)
                and trigger_phrase != ""
            )
        ]

    def filter_completions_by_next_token(self, completions, scores):
        if len(completions) == 0:
            return []
        scores = scores[None, :]
        top_p = 0.95
        top_k = 5
        min_tokens_to_keep = 1
        filter_value = -float("Inf")
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores_processed = scores.masked_fill(indices_to_remove, filter_value)
        probs = torch.nn.functional.softmax(scores_processed, dim=-1)
        next_tokens = torch.topk(probs, top_k)[1][0]
        next_texts = self.tokenizer.convert_ids_to_tokens(next_tokens)

        def is_relevant(completion):
            completion_text = self.completion_text(completion)
            return any([completion_text.startswith(n) for n in next_texts])

        return [completion for completion in completions if is_relevant(completion)]

    def filter_builtin_signatures(self, signatures):
        def is_builtin(signature):
            keyword = self.get_signature_keyword(signature)
            if keyword not in self.signature_cache:
                return False
            completion = self.signature_cache[keyword]
            return completion.detail == "builtins"

        return [signature for signature in signatures if not is_builtin(signature)]

    def check_deprecation_documentation_included(
        self, i, trigger_phrase, current_code: str, deprecated_completions
    ):
        if len(deprecated_completions) == 0:
            return True
        prompt_state_index = i
        if self.beam_tracker.is_beam_search():
            prompt_state_index = self.beam_tracker.get_final_beam_indices()[i]
        prompt_state = self.prompt_states[prompt_state_index]
        comment = prompt_state.get_comment_of_interrupt(DEPRECATION_COMMENT_TYPE)
        if comment is None:
            return False
        # TODO: check if comment actually in line before
        return eq_completions_items(deprecated_completions, comment.context)

    def check_completion_documentation_included(
        self, i: int, trigger_phrase: str, current_code: str, completions
    ):
        # if len(completions) == 0:
        #    return True
        #        if len(completions) < 4:
        #            return True
        prompt_state_index = i
        if self.beam_tracker.is_beam_search():
            prompt_state_index = self.beam_tracker.get_final_beam_indices()[i]
        prompt_state = self.prompt_states[prompt_state_index]
        comment = prompt_state.get_comment_of_interrupt(COMPLETION_COMMENT_TYPE)
        if comment is None:
            if len(completions) == 0:
                return True
            return False
        return eq_completions_items(
            completions, comment.context["completion"]
        ) or in_completion_items(completions, comment.context["completion"])

    def check_signature_documentation_included(
        self, i, trigger_phrase, current_code: str, signature_help
    ):
        if signature_help is None:  # or len(signature_help.signatures) == 0:
            return True
        prompt_state_index = i
        if self.beam_tracker.is_beam_search():
            prompt_state_index = self.beam_tracker.get_final_beam_indices()[i]
        prompt_state = self.prompt_states[prompt_state_index]
        comment = prompt_state.get_comment_of_interrupt(SIGNATURE_COMMENT_TYPE)
        if comment is None:
            if len(signature_help.signatures) == 0:
                return True
            return False

        return eq_signature_help(signature_help, comment.context)

    def should_complete(self, code):
        symbols = [")", "\n", " ", "\t", "]", "}"]
        for symbol in symbols:
            if code.endswith(symbol):
                return False
        return True

    def trigger_interrupt(self, context, interrupt_type_name):
        raise InterruptGeneration(
            interrupt_type_name=interrupt_type_name, context=context
        )

    def get_completion_text(self, completion):
        text = self.completion_text(completion)
        if completion.kind in [
            CompletionItemKind.Function,
            CompletionItemKind.Method,
            CompletionItemKind.Class,
        ]:
            text += "("
        return text

    def get_completions_text(self, completions):
        return [self.get_completion_text(completion) for completion in completions]

    def get_completions_overlap(self, completions: List[str], trigger_phrase: str):
        return [completion.replace(trigger_phrase, "", 1) for completion in completions]

    def get_signature_keyword(self, signature: SignatureInformation) -> str:
        return signature.label.split("(")[0].split("=")[0]

    def increase_signature_cache(self, signature_help: SignatureHelp, completions):
        for completion in completions:
            if not self.is_builtin_completion(completion):
                continue
            keyword = self.completion_text(completion)
            if keyword not in self.signature_cache:
                self.signature_cache[keyword] = completion

    def get_file_name_for_batch(self, i):
        if self.file_names[i // self.expand_size] is not None:
            return self.file_names[i // self.expand_size]
        else:
            return "__generate__.py"

    def scores_for_batch(
        self, i, input_ids: LongTensor, scores: FloatTensor
    ) -> FloatTensor:
        """Returns a 1d FloatTensor for a single batch"""
        current_code = self.current_code(i, input_ids)
        file_name = self.get_file_name_for_batch(i)
        with LspCodeFile(
            file_name, current_code, self.lsp_clients[i // self.expand_size]
        ) as lsp_code_file:
            if self.should_complete(current_code):
                completions = lsp_code_file.ask_completions()
                completions = lsp_code_file.resolve_completions(completions)
            else:
                completions = []
            trigger_phrase = re.search(r"[A-Za-z_]*$", current_code).group(0)
            signature_help = lsp_code_file.ask_signature()
            self.increase_signature_cache(signature_help, completions)
            signature_help.signatures = self.filter_signatures_without_parameters(
                self.filter_builtin_signatures(signature_help.signatures)
            )
            filtered_completions = self.filter_uri(
                self.filter_misc(
                    self.filter_completions_by_postfix(
                        trigger_phrase,
                        self.filter_completions_by_case(
                            trigger_phrase, self.filter_builtin_completions(completions)
                        ),
                    )
                ),
                lsp_code_file.uri,
            )
            filtered_completions.sort(key=lambda x: x.sort_text or x.insert_text)
            # TODO: trigger on selection / Selection INTERRUPT with the SINGULAR completion item
            (
                non_deprecated_completions,
                deprecated_completions,
            ) = self.split_deprecated_completions(filtered_completions)
            deprecated_completions = self.filter_completions_by_next_token(
                deprecated_completions, scores
            )
            if (
                self.config.use_completion_context
                and not self.check_completion_documentation_included(
                    i, trigger_phrase, current_code, non_deprecated_completions
                )
            ):

                if self.config.predict_correct_completion_symbol:

                    if len(non_deprecated_completions) != 1 and (
                        not non_deprecated_completions[0].insert_text.startswith(
                            trigger_phrase
                        )
                        or trigger_phrase == ""
                    ):
                        self.trigger_interrupt(
                            {
                                "completion": non_deprecated_completions,
                                "deprecation": deprecated_completions,
                            },
                            COMPLETION_COMMENT_TYPE,
                        )
                else:
                    self.trigger_interrupt(
                        {"completion": non_deprecated_completions},
                        COMPLETION_COMMENT_TYPE,
                    )
            if (
                self.config.use_deprecation_context
                and not self.check_deprecation_documentation_included(
                    i, trigger_phrase, current_code, deprecated_completions
                )
            ):
                self.trigger_interrupt(deprecated_completions, DEPRECATION_COMMENT_TYPE)
            if (
                self.config.use_signature_context
                and not self.check_signature_documentation_included(
                    i, trigger_phrase, current_code, signature_help
                )
            ):
                self.trigger_interrupt(signature_help, SIGNATURE_COMMENT_TYPE)
            # get completion text for each completion, which may add characters such as ( to functions and , to variables
            # for each completion text, compare to trigger_phrase and get the next few characters
            # add the characters to the current_code to get the next tokens
            (non_deprecated_completion_texts, deprecated_completion_texts) = (
                self.get_completions_text(non_deprecated_completions),
                self.get_completions_text(deprecated_completions),
            )
            (non_deprecated_completion_overlap, deprecated_completion_overlap) = (
                self.get_completions_overlap(
                    non_deprecated_completion_texts, trigger_phrase
                ),
                self.get_completions_overlap(
                    deprecated_completion_texts, trigger_phrase
                ),
            )

            (non_deprecated_first_tokens, deprecated_first_tokens) = (
                self.overlap_unique_first_tokens(
                    current_code, non_deprecated_completion_overlap
                ),
                self.overlap_unique_first_tokens(
                    current_code, deprecated_completion_overlap
                ),
            )
            # if len(non_deprecated_completions) > 0:
            #    c_scores = self.completion_ranker.rank_completions(current_code, non_deprecated_completions)
            #    logger.debug(list(zip(c_scores, [c.label for c in non_deprecated_completions])))
            scores = self.apply_constant_adjustments(
                scores,
                non_deprecated_first_tokens,
                deprecated_first_tokens,
            )
            scores = self.ensure_deprecated_below_non_deprecated(
                scores,
                non_deprecated_first_tokens,
                deprecated_first_tokens,
            )
            if self.config.masked_gen:
                scores = self.mask_other_tokens(scores, non_deprecated_first_tokens)
        return scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        """Returns a 2d FloatTensor which has scores for every batch"""
        if not self.config.lsp_processor or not self.config.enabled:
            return scores
        for i in range(input_ids.shape[0]):
            try:
                scores[i] = self.scores_for_batch(i, input_ids[i], scores[i])
            except InterruptGeneration as ig:
                scores[i][self.interrupt_token_id] = INTERRUPT_LOGITS_SCORE
                self.interrupt = Interrupt(
                    input_ids=input_ids,
                    interrupt_context=ig.context,
                    interrupt_type_name=ig.interrupt_type_name,
                )
                return scores
        return scores

    def resume(self):
        self.interrupt = None
