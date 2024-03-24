from transformers import AutoTokenizer, AutoModel, PreTrainedModel, GenerationMixin
#from llm_lsp.lsp_client import LspClient
from typing import Dict, Any, Optional, List
from llm_lsp.prompt import Prompt
from llm_lsp.message_formatters import MessageFormatter
from llm_lsp.message_formatters.default import DefaultMessageFormatter
from llm_lsp.message_formatters.vanilla import VanillaMessageFormatter
from pygls.lsp.client import BaseLanguageClient
from llm_lsp.interrupts import InterruptType, InterruptStoppingCriteria, Interrupt
from llm_lsp.interrupts.completion import CompletionInterrupt
from llm_lsp.interrupts.signature import SignatureInterrupt
from llm_lsp.lsp import create_lsp_for_language
from llm_lsp.interrupt_mixin import resume
from llm_lsp.lsp.logits_guider import LspLogitsProcessor
from llm_lsp.lsp.boundary_logits_processor import BoundaryLogitsProcessor
from llm_lsp.code_utils import remove_notes, remove_old_notes
import torch.nn.functional as F
import os
import torch
from contextlib import contextmanager

DEFAULT_INTERRUPTS = [
    CompletionInterrupt(),
    SignatureInterrupt()
]

class Generator:
    def __init__(self, model: GenerationMixin, tokenizer: AutoTokenizer, generation_config: Dict[str, Any], message_formatter: MessageFormatter = None, interrupts: List[InterruptType] = DEFAULT_INTERRUPTS, disabled=False):
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        if message_formatter is None:
            message_formatter = VanillaMessageFormatter() if disabled else DefaultMessageFormatter()
        self.message_formatter = message_formatter
        self.interrupts = interrupts
        self.disabled = disabled
        self.init_interrupts()

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples:

        ```python
        # Explicitly ask for tensor allocation on CUDA device :0
        pipe = pipeline(..., device=0)
        with pipe.device_placement():
            # Every framework specific tensor allocation will be done on the request device
            output = pipe(...)
        ```"""
        with torch.device(self.device):
            yield

    def init_interrupts(self):
        interrupt_token_ids = [interrupt.token for interrupt in self.interrupts]
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': interrupt_token_ids
        })
        self.model.resize_token_embeddings(len(self.tokenizer))  
        for interrupt in self.interrupts:
            interrupt.init_input_id(self.tokenizer)

    def remove_padding(self, tokens):
        token_id = self.tokenizer.pad_token_id
        index = (tokens != token_id).nonzero()[0].item()
        return tokens[index:]

    def decode_tokens_remove_interrupt(self, interrupt_token_ids, output_ids):
        if output_ids[-1] in interrupt_token_ids:
            tokens = output_ids[:-1]
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        # Remove eos token
        output_ids = output_ids[:-1]
        if output_ids[-1] in interrupt_token_ids:
            return self.tokenizer.decode(output_ids[:-1], skip_special_tokens=False)
        return self.tokenizer.decode(output_ids, skip_special_tokens=False)

    def handle_completion_generation_result(self, lsp_processor, interrupt_token_ids, output_ids):
        interrupt_id, text = self.decode_tokens_with_maybe_interrupt(
            interrupt_token_ids, output_ids
        )
        # + 1 is for newline added in the prompt creation
        only_generated_code = self.prompt_util.get_whole_code(text)
        if interrupt_id is None:
            only_generated_code = remove_notes(only_generated_code)
            # TODO: code_prefix for generate?
            return True, self.prompt_util.initial_code + "\n" + only_generated_code
        only_generated_code = remove_old_notes(only_generated_code)
        interrupt = [
            interrupt for interrupt in interrupts if interrupt.input_id == interrupt_id
        ][0]
        interrupt_callable = interrupt
        
        prompt = interrupt.edit_generated_code_for_completion(only_generated_code, context)
        return False, prompt

    def pad_input_ids(self, input_ids, edited_input_id):
        # TODO: try remove padding from start
        # TODO: Allow truncation
        pad_token_id = self.tokenizer.pad_token_id
        edited_len = edited_input_id.shape[1]
        inputs_len = input_ids.shape[1]
        pad_to_len = max(edited_len, inputs_len)
        edited_input_id = F.pad(edited_input_id, (pad_to_len-edited_len,0), value=pad_token_id)
        input_ids = F.pad(input_ids, (pad_to_len-inputs_len,0),value=pad_token_id)
        return input_ids, edited_input_id

    def interrupt_input_ids(self):
        return [interrupt.input_id for interrupt in self.interrupts]

    def start_generation(self, prompt, logits_guider, boundary_logits_processor, config):
        stopping_criterium = InterruptStoppingCriteria(self.interrupt_input_ids())

        prompt_input_ids = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids
        generated_sequence = self.model.generate(prompt_input_ids, logits_processor=[logits_guider, boundary_logits_processor], stopping_criteria=[stopping_criterium], **config)
        last_token_ids = generated_sequence[0]
        last_token_ids = self.remove_padding(last_token_ids)
        #last_token_ids = last_token_ids[prompt_input_ids.shape[-1]:]
        decoded_text = self.decode_tokens_remove_interrupt(self.interrupt_input_ids(), last_token_ids)
        return decoded_text

    def resume_generation(self, input_ids, batch_size, logits_guider, boundary_logits_processor, config):
        stopping_criterium = InterruptStoppingCriteria(self.interrupt_input_ids())
        generated_sequences = resume(self.model, input_ids, batch_size, logits_processor=[logits_guider, boundary_logits_processor], stopping_criteria=[stopping_criterium], **config)
        last_token_ids = generated_sequences[0]
        last_token_ids = self.remove_padding(last_token_ids)
        #last_token_ids = last_token_ids[prompt_input_ids.shape[-1]:]
        decoded_text = self.decode_tokens_remove_interrupt(self.interrupt_input_ids(), last_token_ids) 
        return decoded_text   

    def find_interrupt_type(self, interrupt):
        return [i for i in self.interrupts if i.input_id == interrupt.interrupt_token_id][0]

    def edit_generation_text_for_completion(self, decoded_text, prompt_util, interrupt):
        generated_code = prompt_util.get_whole_code(decoded_text)
        generated_code = remove_old_notes(generated_code)

        interrupt_type = self.find_interrupt_type(interrupt)
        edited_generated_code = interrupt_type.edit_generated_code_for_completion(generated_code, interrupt.interrupt_context)
        edited_prompt = prompt_util.format(edited_generated_code)
        return edited_prompt

    def edit_input_ids(self, interrupt, edited_prompt):
        edited_input_ids = self.tokenizer(edited_prompt, return_tensors='pt', add_special_tokens=False).input_ids
        input_ids = interrupt.input_ids
        input_ids, edited_input_ids = self.pad_input_ids(input_ids, edited_input_ids)
        input_ids[interrupt.interrupt_beam] = edited_input_ids
        return input_ids

    def create_lsp_logits_processor(self, lsps, prompt_utils, filenames, expand_size):
        return LspLogitsProcessor(self.tokenizer, lsps, prompt_utils, filenames, expand_size, self.disabled)

    def create_boundary_logits_processor(self):
        return BoundaryLogitsProcessor(self.tokenizer, [".", "("], self.disabled)

    async def complete(self, code: str, repo_root: str, filename: str = "code.py"):
        with self.device_placement():
            return await self._complete(code, repo_root, filename)

    async def _complete(self, code: str, repo_root: str, filename: str = "code.py"):
        batch_size = 1
        # TODO: allow higher batch size
        lsp = await create_lsp_for_language("python", repo_root)
        prompt_util = Prompt(self.tokenizer, self.message_formatter, code)
        prompt_util.init_completion_prompt()
        prompt = prompt_util.format(code)

        config = self.generation_config.copy()
        #if "max_new_tokens" in config:
        #    code_tokens = len(self.tokenizer(code).input_ids)
        #    config["max_new_tokens"] += code_tokens
        expand_size = config["num_beams"] if "num_beams" in config else 1
        logits_guider = self.create_lsp_logits_processor([lsp], [prompt_util], [filename], expand_size)
        boundary_logits_processor = self.create_boundary_logits_processor()
        decoded_text = self.start_generation(prompt, logits_guider, boundary_logits_processor, config)
        if logits_guider.interrupt is None:
            return prompt_util.get_generated_code(decoded_text)
        interrupt = logits_guider.interrupt
        edited_prompt = self.edit_generation_text_for_completion(decoded_text, prompt_util, interrupt)
        input_ids = self.edit_input_ids(interrupt, edited_prompt)
        logits_guider.resume()

        while True:
            decoded_text = self.resume_generation(input_ids, batch_size, logits_guider, boundary_logits_processor, config)
            if logits_guider.interrupt is None:
                result_code = prompt_util.get_generated_code(decoded_text)
                result_code = remove_notes(result_code)
                return result_code
            interrupt = logits_guider.interrupt
            edited_prompt = self.edit_generation_text_for_completion(decoded_text, prompt_util, interrupt)
            input_ids = self.edit_input_ids(interrupt, edited_prompt)
            logits_guider.resume()