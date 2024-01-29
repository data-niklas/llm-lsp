from llm_lsp.lsp_processor import LspLogitsProcessor
from pygls.lsp.client import BaseLanguageClient
import asyncio
import argparse
from transformers import AutoTokenizer, Pipeline
import transformers
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer
import torch
from logzero import logger
from lsprotocol.types import *
import tempfile
import shutil
import os
from os import path
from llm_lsp.lsp_client import LspClient

# https://github.com/swyddfa/lsp-devtools/blob/develop/lib/pytest-lsp/pytest_lsp/clients/visual_studio_code_v1.65.2.json

import nest_asyncio

nest_asyncio.apply()


MODEL = "codellama/CodeLlama-7b-Instruct-hf"

PROMPT_TEMPLATE = "You are a code completion tool. Complete the following Python tool. Only provide the completed code. Do not return descriptions of your actions. Do not generate more code than necessary."
GLOBAL_CONFIGURATION = {
    #"num_beam_groups": 2,
    "num_beams": 2,
    #"diversity_penalty": 1.0,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1,
    "return_full_text": False,
    "temperature": 0.7,
    "max_new_tokens": 2048,
}


def highlight_code(code):
    return highlight(code, PythonLexer(), Terminal256Formatter())


def find_venv(root):
    dir = path.join(root, "venv")
    if path.exists(dir):
        return dir
    # TODO: add more


async def main(args):
    lsp_client = LspClient()
    await lsp_client.start("pylsp", [])
    lsp: BaseLanguageClient = BaseLanguageClient("pylsp", "1.0.0")
    await lsp.start_io("pylsp")
    logger.debug("Now initializing")

    @lsp.feature("workspace/configuration")
    def configuration(ls, params):
        logger.debug("It wants configuration!!!")

    @lsp.feature("textDocument/publishDiagnostics")
    def diagnostics(ls, params):
        #pass
        diagnostics = [d for d in params.diagnostics if d.tags and DiagnosticTag.Deprecated in d.tags]
        if len(diagnostics) > 0:
            logger.debug(diagnostics)
        #logger.debug(params)

    initialize_result = await lsp.initialize_async(
        InitializeParams(
            root_path=args.directory,
            capabilities=ClientCapabilities(
                
                workspace=WorkspaceClientCapabilities(
                    configuration=True,
                    did_change_configuration=DidChangeConfigurationClientCapabilities(
                        dynamic_registration=True
                    ),
                    workspace_folders=True
                ),
                text_document=TextDocumentClientCapabilities(
                    completion=CompletionClientCapabilities(
                        completion_item=CompletionClientCapabilitiesCompletionItemType(
                            snippet_support=True,
                            deprecated_support=True,
                            documentation_format=["markdown", "plaintext"],
                            preselect_support=True,
                            label_details_support=True,
                            resolve_support=CompletionClientCapabilitiesCompletionItemTypeResolveSupportType(
                                properties=[
                                    "deprecated",
                                    "documentation",
                                    "detail",
                                    "additionalTextEdits",
                                ]
                            ),
                            tag_support=CompletionClientCapabilitiesCompletionItemTypeTagSupportType(value_set=[
                                CompletionItemTag.Deprecated
                            ])
                        )
                    ),
                    publish_diagnostics=PublishDiagnosticsClientCapabilities(
                        tag_support=PublishDiagnosticsClientCapabilitiesTagSupportType(
                            value_set=[DiagnosticTag.Deprecated]
                        )
                    )
                ),
            ),
        )
    )
    # logger.debug(initialize_result)

    lsp.initialized(InitializedParams())
    logger.debug(
        f"Using python: "
        + path.abspath(path.join(find_venv(args.directory), "bin", "python"))
    )
    lsp.workspace_did_change_configuration(
        DidChangeConfigurationParams(
            settings={
                "pylsp.plugins.jedi.environment": path.abspath(
                    path.join(find_venv(args.directory), "bin", "python")
                ),
                "pylsp.plugins.jedi_completion.include_class_objects": True,
                "pylsp.plugins.jedi_completion.include_function_objects": True,
                "pylsp.plugins.rope_completion.enabled": True,
            }
        )
    )
    #from time import sleep
    #sleep(15)  # Wait for initialization
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pipeline: Pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    code = ""
    with open(args.file, "r") as f:
        code = f.read()
    prompt = "[INST] " + PROMPT_TEMPLATE + "\n[/INST]\n" + code
    processor = LspLogitsProcessor(tokenizer, lsp, len(prompt), args.file, code)

    def create_prompt():
        yield prompt

    for sequences in pipeline(
        create_prompt(),
        eos_token_id=tokenizer.eos_token_id,
        logits_processor=[processor],
        **GLOBAL_CONFIGURATION,
    ):
        texts = [sequence["generated_text"] for sequence in sequences]
        hl = highlight_code(code + texts[0])
        hl = code + texts[0]
        logger.debug("Code:\n##########\n" + hl + "\n##########")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Pseudo code meta strategy",
        description="Stuff",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--file")
    parser.add_argument("-d", "--directory")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
