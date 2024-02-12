from pygls.lsp.client import BaseLanguageClient
import asyncio
import argparse
from transformers import AutoTokenizer, Pipeline, LogitsProcessorList, StoppingCriteriaList, StoppingCriteria
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
from llm_lsp.lsp_processor import LspLogitsProcessor
from llm_lsp.lsp_client import LspClient
from llm_lsp.constants import *
from typing import Tuple
import subprocess
from llm_lsp.interrupts import InterruptStoppingCriteria, decode_tokens_with_maybe_interrupt, Interrupt, get_new_prompt_or_finish, handle_deprecation_interrupt, handle_signature_interrupt

# https://github.com/swyddfa/lsp-devtools/blob/develop/lib/pytest-lsp/pytest_lsp/clients/visual_studio_code_v1.65.2.json

import nest_asyncio

nest_asyncio.apply()



# 1. meta-strategy part-by-part generation
# -> pipeline -> add information to prompt -> pipeline -> add information to prompt -> pipeline -> done
# custom token in vocab
# add comment so that first part is not 

# 2. token by token
# -> pipeline -> done
# (using LLM to rerank output completion list)

# 


def highlight_code(code):
    return highlight(code, PythonLexer(), Terminal256Formatter())


def find_venv(root):
    dir = path.join(root, "venv")
    if path.exists(dir):
        return dir
    # TODO: add more

def add_special_tokens(pipeline, tokenizer, interrupt_token_ids):
    tokenizer.add_special_tokens({
        'additional_special_tokens': interrupt_token_ids
    })
    pipeline.model.resize_token_embeddings(len(tokenizer))

def initialize_generation(interrupts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pipeline: Pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    interrupt_token_ids = [interrupt.token for interrupt in interrupts]
    add_special_tokens(pipeline, tokenizer, interrupt_token_ids)
    return pipeline, tokenizer


def generate_code(pipeline, tokenizer, lsp, args, code, interrupts):
    for interrupt in interrupts:
        interrupt.input_id = tokenizer.convert_tokens_to_ids(interrupt.token)
    interrupt_input_ids = [interrupt.input_id for interrupt in interrupts]
    prompt = "[INST] " + PROMPT_TEMPLATE + "\n[/INST]\n" + code
    text_len_prompt_with_initial_code = len(prompt)
    processor = LspLogitsProcessor(tokenizer, lsp, len(prompt), args.file, code, pipeline)
    while True:
        sequences = pipeline(
            prompt,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([processor]),
            stopping_criteria=StoppingCriteriaList([InterruptStoppingCriteria(interrupt_input_ids)]),
            return_tensors=True,
            **GLOBAL_CONFIGURATION,
        )
        last = sequences[-1]
        last_token_ids = last["generated_token_ids"]
        finished, text = get_new_prompt_or_finish(tokenizer, interrupts, last_token_ids, text_len_prompt_with_initial_code, processor, code)
        if finished:
            return text
        prompt = text

async def main(args):
    interrupts = [
        Interrupt(token=DEPRECATION_INTERRUPT_TOKEN, callable=handle_deprecation_interrupt),
        Interrupt(token=SIGNATURE_INTERRUPT_TOKEN, callable=handle_signature_interrupt)
    ]
    logger.setLevel(args.level)
    lsp_client = LspClient()
    await lsp_client.start("pylsp", [])
    lsp: BaseLanguageClient = BaseLanguageClient("pylsp", "1.0.0")
    await lsp.start_io("pylsp")
    logger.info("Now initializing")

    @lsp.feature("workspace/configuration")
    def configuration(ls, params):
        logger.debug("It wants configuration!!!")

    @lsp.feature("textDocument/publishDiagnostics")
    def diagnostics(ls, params):
        # pass
        diagnostics = [
            d
            for d in params.diagnostics
            if d.tags and DiagnosticTag.Deprecated in d.tags
        ]
        if len(diagnostics) > 0:
            logger.debug(diagnostics)
        # logger.debug(params)

    initialize_result = await lsp.initialize_async(
        InitializeParams(
            root_path=args.directory,
            capabilities=ClientCapabilities(
                workspace=WorkspaceClientCapabilities(
                    configuration=True,
                    did_change_configuration=DidChangeConfigurationClientCapabilities(
                        dynamic_registration=True
                    ),
                    workspace_folders=True,
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
                            tag_support=CompletionClientCapabilitiesCompletionItemTypeTagSupportType(
                                value_set=[CompletionItemTag.Deprecated]
                            ),
                        )
                    ),
                    publish_diagnostics=PublishDiagnosticsClientCapabilities(
                        tag_support=PublishDiagnosticsClientCapabilitiesTagSupportType(
                            value_set=[DiagnosticTag.Deprecated]
                        )
                    ),
                ),
            ),
        )
    )
    # logger.debug(initialize_result)

    lsp.initialized(InitializedParams())
    logger.info(
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
    # from time import sleep
    # sleep(15)  # Wait for initialization
    pipeline, tokenizer = initialize_generation(interrupts)
    code = ""
    with open(args.file, "r") as f:
        code = f.read()
    code = generate_code(pipeline, tokenizer, lsp, args, code, interrupts)
    hl = highlight_code(code)
    #hl = code + texts[0]
    logger.info("Code:\n##########\n" + hl + "\n##########")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Pseudo code meta strategy",
        description="Stuff",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--file")
    parser.add_argument("-d", "--directory")
    parser.add_argument("-l", "--level", default="DEBUG")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
