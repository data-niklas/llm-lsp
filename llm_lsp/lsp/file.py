import uuid
from os import path
import os
import asyncio
from lsprotocol.types import *


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
        if len(completions) > 0:
            resolved_completions = asyncio.get_event_loop().run_until_complete(
                asyncio.gather(
                    *[
                        self.lsp_client.completion_item_resolve_async(completion)
                        for completion in completions
                    ]
                )
            )
        else:
            resolved_completions = []
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
        if code == "":
            return 0, 0
        lines = code.splitlines()
        last_line_index = len(lines) - 1
        last_line = lines[last_line_index]
        last_char_index = len(last_line)
        return max(last_char_index, 0), max(last_line_index, 0)
