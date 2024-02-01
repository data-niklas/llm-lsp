from pygls.lsp.client import BaseLanguageClient
import asyncio
import argparse
from transformers import AutoTokenizer, Pipeline
import transformers
import torch
from logzero import logger
import os
from os import path
import sys

MODEL = "codellama/CodeLlama-7b-Instruct-hf"

async def main():
    #from time import sleep
    #sleep(15)  # Wait for initialization
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    logger.debug(tokenizer.decode([int(sys.argv[1])]))
    if True:
        return
    words = " ".join(sys.argv[1:])
    ids = tokenizer(words, add_special_tokens=False).input_ids
    logger.debug(ids)
    logger.debug(tokenizer.batch_decode(ids))




if __name__ == "__main__":
    asyncio.run(main())
