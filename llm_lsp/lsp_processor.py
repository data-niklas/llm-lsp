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


class LspLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, lsp_client, prompt_len, file, code):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.lsp_client: BaseLanguageClient = lsp_client
        self.prompt_len = prompt_len
        self.file = file
        self.code = code

    def char_line_of_code(self, code):
        lines = code.splitlines()
        last_line_index = len(lines) - 1
        last_line = lines[last_line_index]
        last_char_index = len(last_line)
        return max(last_char_index, 0), max(last_line_index, 0)

    def create_temp_file(self):
        return self.file + "_" + str(uuid.uuid1())

    def ask_completion(self, input_ids: LongTensor):
        decoded_input = self.tokenizer.batch_decode(input_ids)[0]
        code_with_completion = self.code + decoded_input[self.prompt_len :]
        temp_file = self.create_temp_file()
        with open(temp_file, "w") as f:
            f.write(code_with_completion)
        self.lsp_client.text_document_did_open(
            DidOpenTextDocumentParams(
                text_document=TextDocumentItem(
                    uri="file://" + path.abspath(temp_file),
                    language_id="python",
                    version=1,
                    text=code_with_completion,
                )
            )
        )
        char, line = self.char_line_of_code(code_with_completion)
        awaitable = self.lsp_client.text_document_completion_async(
            CompletionParams(
                text_document=TextDocumentIdentifier(
                    uri="file://" + path.abspath(temp_file)
                ),
                position=Position(character=char, line=line),
                context=CompletionContext(
                    trigger_kind=CompletionTriggerKind.Invoked, trigger_character="."
                ),
            )
        )
        items = asyncio.get_event_loop().run_until_complete(awaitable)
        if isinstance(items, CompletionList):
            items = items.items
        items = [
            asyncio.get_event_loop().run_until_complete(
                self.lsp_client.completion_item_resolve_async(item)
            )
            for item in items
        ]
        return items, code_with_completion, temp_file

    def clean(self, temp_file):
        self.lsp_client.text_document_did_close(
            DidCloseTextDocumentParams(
                TextDocumentIdentifier(uri="file://" + path.abspath(temp_file))
            )
        )
        os.remove(temp_file)

    def ids_to_text(self, input_ids: LongTensor) -> str:
        return self.tokenizer.decode(input_ids)

    def current_code(self, input_ids: LongTensor) -> str:
        """The complete generated code at this point. This includes the starting code and the generated code."""
        generated_code_with_prompt = self.ids_to_text(input_ids)
        generated_code = generated_code_with_prompt[self.prompt_len :]
        return self.code + generated_code

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
                CompletionItemKind.Function,
                CompletionItemKind.Property,
            ]
        ]

    def filter_builtin_completions(self, completions):
        return [
            completion for completion in completions if completion.detail != "builtins"
        ]

    def filter_misc(self, completions):
        return [
            completion
            for completion in completions
            if not self.completion_text(completion).startswith("__")
        ]

    def is_deprecated(self, completion) -> bool:
        return "deprecated" in completion.documentation.value

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
            # logger.debug(f"Searching for {input_ids[-1]} in {tokens}")
            last_input_id_index = tokens.index(input_ids[-1])
            # logger.debug(tokens)
            # logger.debug(input_ids[-5:])
            for i in reversed(range(last_input_id_index)):
                if input_ids[input_ids_index] != tokens[i]:
                    return -1
                input_ids_index -= 1
            text = self.tokenizer.decode(input_ids[-min(len(input_ids), 6) :])
            logger.debug(f"An index of {last_input_id_index} has occured for '{text}'")
            return last_input_id_index
        except ValueError:
            return -1
        except IndexError:
            return -1

    def get_unique_first_tokens(self, indexes_after_overlap, tokens):
        result_tokens = []
        for index, tokens in zip(indexes_after_overlap, tokens):
            if index >= len(tokens):
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

    def downrank_completed_items(self, scores, non_deprecated_index_after_last_overlapping_token, non_deprecated_tokens):
        non_deprecated_unique_first_tokens = self.get_unique_first_tokens(non_deprecated_index_after_last_overlapping_token, non_deprecated_tokens)
        for token in non_deprecated_unique_first_tokens:
            scores[token] -= 14
        return scores

    def scores_for_batch(
        self, input_ids: LongTensor, scores: FloatTensor
    ) -> FloatTensor:
        """Returns a 1d FloatTensor for a single batch"""
        current_code = self.current_code(input_ids)
        # logger.debug("CODE")
        # logger.debug(current_code)
        with LspCodeFile(self.file, current_code, self.lsp_client) as lsp_code_file:
            resolved_completions = lsp_code_file.ask_completions()
            filtered_completions = self.filter_misc(
                self.filter_completions_by_kind(
                    self.filter_builtin_completions(resolved_completions)
                )
            )
            (
                non_deprecated_completions,
                deprecated_completions,
            ) = self.split_deprecated_completions(filtered_completions)
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
            # TODO: determine if done
            if self.count_full_match(
                non_deprecated_index_after_last_overlapping_token, non_deprecated_tokens
            ) == 1:
                # Just assuming that the deprecated items are irrelevant, as they should have been downranked quite heavily
                logger.debug("Found a *single* full match so returning early")
                # And now downrank items, as they should not occur, as the completion item has been fully used
                return self.downrank_completed_items(scores, non_deprecated_index_after_last_overlapping_token, non_deprecated_tokens)
            (
                non_deprecated_unique_first_tokens,
                deprecated_unique_first_tokens,
            ) = self.get_unique_first_tokens(
                non_deprecated_index_after_last_overlapping_token, non_deprecated_tokens
            ), self.get_unique_first_tokens(
                deprecated_index_after_last_overlapping_token, deprecated_tokens
            )
            for non_deprecated_token in non_deprecated_unique_first_tokens:
                scores[non_deprecated_token] += 7.0
            for deprecated_token in deprecated_unique_first_tokens:
                scores[deprecated_token] -= 7.0
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

    def char_line_of_code(self, code):
        lines = code.splitlines()
        last_line_index = len(lines) - 1
        last_line = lines[last_line_index]
        last_char_index = len(last_line)
        return max(last_char_index, 0), max(last_line_index, 0)