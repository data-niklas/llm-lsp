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
from llm_lsp.interrupts.completion import is_deprecated
from llm_lsp.interrupts import Interrupt
from llm_lsp.lsp.file import LspCodeFile
from dataclasses import dataclass
from typing import Any


@dataclass
class InterruptGeneration(BaseException):
    interrupt_token_id: int
    context: Any


def custom_completion_item_hash(self):
    return hash((self.label, self.kind))


CompletionItem.__hash__ = custom_completion_item_hash


class LspLogitsProcessor(LogitsProcessor):
    # TODO: Filter class completions, if they have no prefix in the code yet, they should not randomly influence results if the model on its down does not decide to maybe use it
    # This should stop Parser(fields=Parser)
    def __init__(self, tokenizer, lsp_clients, prompt_utils, filenames, expand_size, disabled):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.lsp_clients: BaseLanguageClient = lsp_clients
        self.prompt_utils = prompt_utils
        self.filenames = filenames
        self.signature_cache = {}
        self.expand_size = expand_size
        self.interrupt = None
        self.disabled = disabled

    def ids_to_text(self, input_ids: LongTensor) -> str:
        # remove padding
        token_id = self.tokenizer.pad_token_id
        index = (input_ids != token_id).nonzero()[0].item()
        input_ids = input_ids[index:]
        return self.tokenizer.decode(input_ids, skip_special_tokens=False)

    def current_code(self, i, input_ids: LongTensor) -> str:
        """The complete generated code at this point. This includes the starting code and the generated code."""
        generated_code_with_prompt = self.ids_to_text(input_ids)
        code = self.prompt_utils[i // self.expand_size].get_whole_code(
            generated_code_with_prompt
        )
        return code

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
                CompletionItemKind.Variable,
            ]
        ]

    def filter_builtin_completions(self, completions):
        return [
            completion for completion in completions if completion.detail != "builtins"
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
            scores[token] = scores[token] - 14
        return scores

    def apply_constant_adjustments(
        self, scores, non_deprecated_unique_first_tokens, deprecated_unique_first_tokens
    ):
        for non_deprecated_token in non_deprecated_unique_first_tokens:
            scores[non_deprecated_token] += 7.0
        for deprecated_token in deprecated_unique_first_tokens:
            scores[deprecated_token] -= 7.0
        return scores

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

    def filter_builtin_signatures(self, signatures):
        def is_builtin(signature):
            keyword = self.get_signature_keyword(signature)
            if keyword not in self.signature_cache:
                return False
            completion = self.signature_cache[keyword]
            return completion.detail == "builtins"

        return [signature for signature in signatures if not is_builtin(signature)]

    def uprank_divider_after_completion(self, scores, input_ids):
        text = self.tokenizer.decode(input_ids[-1:])
        open_token = self.tokenizer(text + "(", add_special_tokens=False).input_ids[-1]
        scores[open_token] += 14
        return scores

    def check_deprecation_documentation_included(
        self, current_code: str, deprecated_completions
    ):
        if len(deprecated_completions) == 0:
            return True
        return "# Deprecation note: " in current_code
        # lines = current_code.splitlines()[:-1]
        # lines.reverse()
        # for line in lines:
        #     line = line.strip()
        #     if not line.startswith("# "):
        #         return False
        #     elif line.startswith("# Deprecation note: "):
        #         return True
        # return False

    def check_signature_documentation_included(self, current_code: str, signature_help):
        if signature_help is None or len(signature_help.signatures) == 0:
            return True
        current_code = current_code.rstrip(" \t\n")
        first_signature = signature_help.signatures[
            signature_help.active_signature
        ].label
        return "# Signature note: " in current_code
        # lines = current_code.splitlines()[:-1]
        # lines.reverse()
        # for line in lines:
        #     line = line.strip()
        #     if not line.startswith("# "):
        #         return False
        #     elif line.startswith("# Signature note: "):
        #         return True
        # return False

    def should_complete(self, code):
        symbols = [")", "\n", " ", "\t", "]", "}"]
        for symbol in symbols:
            if code.endswith(symbol):
                return False
        return True

    def trigger_interrupt(self, context, j):
        input_id = self.tokenizer.convert_tokens_to_ids(j)
        raise InterruptGeneration(interrupt_token_id=input_id, context=context)

    def get_completion_text(self, completion):
        text = self.completion_text(completion)
        if completion.kind in [CompletionItemKind.Method, CompletionItemKind.Class]:
            text += "("
        return text

    def downrank_comments(self, current_code, scores):
        hashtag_id = self.tokenizer(current_code + "#", add_special_tokens=False).input_ids[-1]
        scores[hashtag_id] = scores[hashtag_id] - 20
        return scores

    def get_completions_text(self, completions):
        return [self.get_completion_text(completion) for completion in completions]

    def get_completions_overlap(self, completions: List[str], trigger_phrase: str):
        return [completion.replace(trigger_phrase, "", 1) for completion in completions]

    def get_signature_keyword(self, signature: SignatureInformation) -> str:
        return signature.label.split("(")[0].split("=")[0]

    def increase_signature_cache(self, signature_help: SignatureHelp, completions):
        signatures = signature_help.signatures
        for signature in signatures:
            # TODO: replace string splitting with parsing per language
            keyword = self.get_signature_keyword(signature)
            if keyword in self.signature_cache:
                continue
            relevant_completions = [
                completion
                for completion in completions
                if self.get_completion_text(completion) == keyword
            ]
            if len(relevant_completions) == 0:
                continue
            self.signature_cache[keyword] = relevant_completions[0]

    def filename(self, i):
        if self.filenames[i // self.expand_size] is not None:
            return self.filenames[i // self.expand_size]
        else:
            return "__generate__.py"

    def scores_for_batch(
        self, i, input_ids: LongTensor, scores: FloatTensor
    ) -> FloatTensor:
        """Returns a 1d FloatTensor for a single batch"""
        current_code = self.current_code(i, input_ids)
        filename = self.filename(i)
        with LspCodeFile(
            filename, current_code, self.lsp_clients[i // self.expand_size]
        ) as lsp_code_file:
            if self.should_complete(current_code):
                resolved_completions = lsp_code_file.ask_completions()
            else:
                resolved_completions = []
            trigger_phrase = re.search(r"[A-Za-z_]*$", current_code).group(0)
            signature_help = lsp_code_file.ask_signature()
            self.increase_signature_cache(signature_help, resolved_completions)
            signature_help.signatures = self.filter_builtin_signatures(
                signature_help.signatures
            )
            filtered_completions = self.filter_misc(
                self.filter_completions_by_postfix(
                    trigger_phrase,
                    self.filter_completions_by_case(
                        trigger_phrase,
                        self.filter_completions_by_kind(
                            self.filter_builtin_completions(resolved_completions)
                        ),
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
                self.trigger_interrupt(deprecated_completions, "[COMPLETION_INTERRUPT]")
            if not self.check_signature_documentation_included(
                current_code, signature_help
            ):
                self.trigger_interrupt(signature_help, "[SIGNATURE_INTERRUPT]")
                # TODO: do not interrupt on stdlib stuff

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
            scores = self.downrank_comments(current_code, scores)
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
        return scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        """Returns a 2d FloatTensor which has scores for every batch"""
        if self.disabled:
            return scores
        for i in range(input_ids.shape[0]):
            try:
                scores[i] = self.scores_for_batch(i, input_ids[i], scores[i])
            except InterruptGeneration as ig:
                scores[i][ig.interrupt_token_id] = 100
                self.interrupt = Interrupt(
                    input_ids=input_ids,
                    interrupt_beam=i,
                    interrupt_context=ig.context,
                    interrupt_token_id=ig.interrupt_token_id,
                )
                return scores
        return scores

    def resume(self):
        self.interrupt = None