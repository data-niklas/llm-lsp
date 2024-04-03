from pygls.lsp.client import BaseLanguageClient
import asyncio
import argparse
from transformers import AutoTokenizer, Pipeline, LogitsProcessorList, StoppingCriteriaList, StoppingCriteria, AutoModel
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
from llm_lsp.constants import *
from typing import Tuple
import subprocess
from llm_lsp.generator import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM
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



def highlight_code(code):
    return highlight(code, PythonLexer(), Terminal256Formatter())


async def main(args):
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.half().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_config = GLOBAL_CONFIGURATION
    generator = Generator(model, tokenizer, generation_config)
    if args.strategy == "COMPLETE":
        with open(args.file, "r") as f:
            code = f.read()
        repo_root = args.directory
        filename = args.file
        completed_code = await generator.complete(code, repo_root, filename)
        code += completed_code
        hl = highlight_code(code)
        logger.info("Code:\n##########\n" + hl + "\n##########")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Pseudo code meta strategy",
        description="Stuff",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--file", default="tests/pydantic_3.py")
    parser.add_argument("-d", "--directory", default=".")
    parser.add_argument("-l", "--level", default="DEBUG")
    parser.add_argument("-s", "--strategy", default="COMPLETE")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
