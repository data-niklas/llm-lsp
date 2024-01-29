from transformers import LogitsProcessor, PreTrainedTokenizer

from torch import LongTensor, FloatTensor, full_like
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

    def __call__(self, input_ids: LongTensor, scores: FloatTensor):
        items, code, temp_file = self.ask_completion(input_ids)
        # if len(code) > 0 and code[-1] != ".":
        #    self.clean(temp_file)
        #    return scores
        # for item in items:
        # logger.debug((item.insert_text or item.label, item.kind))
        old_items = items
        items = [
            item
            for item in items
            if item.kind
            in [
                CompletionItemKind.Method,
                CompletionItemKind.Field,
                CompletionItemKind.Function,
                CompletionItemKind.Property,
            ]
            and item.detail
            != "builtins"  # Filter builtins, such that hopefully only methods remain
        ]

        for item in items:
            if (item.insert_text or item.label).startswith(code.split(".")[-1]) and len(code.split(".")[-1]) > 2 and "deprecated" in item.documentation.value:
                logger.debug("NUKE THE RESULT")
                logger.debug(code)
                #return full_like(scores, -20)

        items = [
            item
            for item in items
            if not code.endswith(item.insert_text or item.label)
        ]

        if len(items) == 0:
            logger.debug("NO ITEMS FOR CODE:")
            logger.debug([item.insert_text or item.label for item in old_items])
            logger.debug(code)
            if [item.insert_text or item.label for item in old_items] == ["dict"]:
                logger.debug(old_items)

        deprecated_items = [
            item for item in items if "deprecated" in item.documentation.value
        ]  # simulate deprecated check]

        items = [
            item for item in items if "deprecated" not in item.documentation.value
        ]  # simulate deprecated check
        items_text = [
            item.insert_text if item.insert_text else item.label for item in items
        ]
        deprecated_items_text = [
            item.insert_text if item.insert_text else item.label
            for item in deprecated_items
        ]
        if len(items_text) > 0 or len(deprecated_items_text) > 0:
            logger.debug(code)
        if len(items_text) > 0:
            logger.debug("UPGRADED:")
            logger.debug(items_text)
        if len(deprecated_items_text) > 0:
            logger.debug("DEPRECATED:")
            logger.debug(deprecated_items_text)
        # TODO: filtering and sorting
        items_tokens = [
            self.tokenizer(item, add_special_tokens=False).input_ids
            for item in items_text
        ]
        deprecated_items_tokens = [
            self.tokenizer(item, add_special_tokens=False).input_ids
            for item in deprecated_items_text
        ]
        first_input_ids = list(set([item[0] for item in items_tokens if len(item) > 0]))
        deprecated_first_input_ids = list(
            set([item[0] for item in deprecated_items_tokens if len(item) > 0])
        )
        for input_id in first_input_ids:
            # First dimension is batchsize, second is vocab
            # logger.debug(scores[0][input_id])
            scores[0][input_id] += 14
        for input_id in deprecated_first_input_ids:
            # First dimension is batchsize, second is vocab
            # logger.debug(scores[0][input_id])
            scores[0][input_id] -= 21

        self.clean(temp_file)
        return scores
