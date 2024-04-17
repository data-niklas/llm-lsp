from transformers import AutoTokenizer, AutoModel, PreTrainedModel, GenerationMixin
from transformers.generation import GenerationMode, GenerationConfig
import warnings
from tree_sitter import Language, Parser
import tree_sitter_python
PY_LANGUAGE = Language(tree_sitter_python.language(), "python")
# from llm_lsp.lsp_client import LspClient
from typing import Dict, Any, Optional, List
from llm_lsp.prompt import Prompt
from llm_lsp.message_formatters import MessageFormatter
from llm_lsp.message_formatters.default import DefaultMessageFormatter
from llm_lsp.message_formatters.vanilla import VanillaMessageFormatter
from pygls.lsp.client import BaseLanguageClient
from llm_lsp.interrupts import InterruptType, InterruptStoppingCriteria, Interrupt
from llm_lsp.interrupts.deprecation import DeprecationInterrupt
from llm_lsp.interrupts.signature import SignatureInterrupt
from llm_lsp.lsp import create_lsp_for_language
from llm_lsp.interrupt_mixin import resume
from llm_lsp.lsp.logits_guider import LspLogitsProcessor
from llm_lsp.lsp.boundary_logits_processor import BoundaryLogitsProcessor
from llm_lsp.lsp.comments_processor import CommentsLogitsProcessor
from llm_lsp.code_utils import CodeUtil
from llm_lsp.code_utils.python import PythonCodeUtil
from llm_lsp.commentor import Commentor, Comment
import torch.nn.functional as F
import os
import torch
from contextlib import contextmanager
from copy import deepcopy
import copy
import time

DEFAULT_INTERRUPTS = [DeprecationInterrupt(), SignatureInterrupt()]

PAD_TOKEN = "[PAD]"


class Generator:
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: AutoTokenizer,
        generation_config: Dict[str, Any],
        message_formatter: MessageFormatter = None,
        interrupts: List[InterruptType] = DEFAULT_INTERRUPTS,
        disabled=False,
    ):
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        if message_formatter is None:
            message_formatter = (
                VanillaMessageFormatter() if disabled else DefaultMessageFormatter()
            )
        self.message_formatter = message_formatter
        self.interrupts = interrupts
        self.disabled = disabled
        self.add_special_tokens()

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

    def add_special_tokens(self):
        interrupt_token_ids = [interrupt.token for interrupt in self.interrupts]
        additional_special_tokens = interrupt_token_ids + [PAD_TOKEN]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(PAD_TOKEN)
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

    def handle_completion_generation_result(
        self, lsp_processor, interrupt_token_ids, output_ids
    ):
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

        prompt = interrupt.edit_generated_code_for_completion(
            only_generated_code, context
        )
        return False, prompt

    def pad_input_ids(self, input_ids, edited_input_id):
        # TODO: try remove padding from start
        # TODO: Allow truncation
        pad_token_id = self.tokenizer.pad_token_id
        edited_len = edited_input_id.shape[1]
        inputs_len = input_ids.shape[1]
        pad_to_len = max(edited_len, inputs_len)
        edited_input_id = F.pad(
            edited_input_id, (pad_to_len - edited_len, 0), value=pad_token_id
        )
        input_ids = F.pad(input_ids, (pad_to_len - inputs_len, 0), value=pad_token_id)
        return input_ids, edited_input_id

    def interrupt_input_ids(self):
        return [interrupt.input_id for interrupt in self.interrupts]

    def expand_input_ids(self, input_ids, config):
        if (
            self.model.generation_config._from_model_config
            and self.model.generation_config._original_object_hash
            == hash(self.model.generation_config)
            and self.model.config._has_non_default_generation_parameters()
        ):
            new_generation_config = GenerationConfig.from_model_config(
                self.model.config
            )
            if new_generation_config != self.model.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                self.model.generation_config = new_generation_config
        generation_config = self.model.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(
            **config
        )  # All unused kwargs must be model kwargs
        generation_mode = generation_config.get_generation_mode(None)
        if generation_mode in [
            GenerationMode.ASSISTED_GENERATION,
            GenerationMode.GREEDY_SEARCH,
            GenerationMode.CONTRASTIVE_SEARCH,
        ]:
            return input_ids, model_kwargs
        elif generation_mode == GenerationMode.SAMPLE:
            return self.model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.SAMPLE:
            return self.model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )
        elif generation_mode in [
            GenerationMode.BEAM_SAMPLE,
            GenerationMode.BEAM_SEARCH,
            GenerationMode.GROUP_BEAM_SEARCH,
            GenerationMode.CONSTRAINED_BEAM_SEARCH,
        ]:
            return self.model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )

    def track_beam_selections(self, beam_indices, commentors, code_utils):
        """Beam indices for each batch. Access the selected beams using batch_index * batch_size + beam_index_in_batch

        Args:
            beam_indices (_type_): _description_
        """
        # TODO: add batching
        new_commentors = []
        new_codeutils = []
        indices = list(range(len(commentors)))
        change_count = len(beam_indices[0])
        beam_batch_count = len(beam_indices)
        for i in range(change_count):
            new_indices = deepcopy(indices)
            for j in range(beam_batch_count):
                tuple_index = beam_indices[j]
                torch_index = tuple_index[i]
                index = torch_index.item()
                new_indices[j] = indices[index]
            indices = new_indices

        # For debugging, will be removed when not needed anymore, so probably never
        a = 1

        # I miss you already, my small a

        for i in range(len(indices)):
            index = indices[i]
            commentor = commentors[index]
            code_util = code_utils[index]
            new_commentors.append(deepcopy(commentor))
            new_codeutils.append(deepcopy(code_util))

        # Change in place
        for i in range(len(beam_indices)):
            commentors[i] = new_commentors[i]
            code_utils[i] = new_codeutils[i]

    def index_of_beam_to_edit(self, input_ids):
        eos_tokens = self.interrupt_input_ids()
        if isinstance(self.tokenizer.eos_token_id, int):
            eos_tokens.append(self.tokenizer.eos_token_id)
        else:
            eos_tokens += self.tokenizer.eos_token_id
        final_tokens = input_ids[:, -1:].view(-1)
        result = sum(final_tokens == token_id for token_id in eos_tokens).nonzero()
        if len(result) == 0:
            # Return 0 when e.g. the maximum length of input_ids has been reached and thus no interrupt has been found
            return 0
        return result.item()

    def resume_generation(
        self,
        input_ids,
        batch_size,
        logits_guider,
        boundary_logits_processor,
        comments_logits_processor,
        config,
        commentors,
        code_utils,
    ):
        """
        Returns the decoded text for each batch. If using beam search this is the decoded text of the best beam. In addition this method will return the index of the best beam. This is necessary to access the appropriate comment tool.
        """
        stopping_criterium = InterruptStoppingCriteria(self.interrupt_input_ids())
        logits_processor = [logits_guider, boundary_logits_processor, comments_logits_processor]
        kwargs = {}
        if "pad_token_id" not in config:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        generated_result = resume(
            self.model,
            input_ids,
            batch_size,
            logits_processor=logits_processor,
            stopping_criteria=[stopping_criterium],
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
            **config,
        )
        generated_sequences = generated_result["sequences"]

        if "beam_input_ids" in generated_result and logits_guider.interrupt is not None:
            beam_input_ids = generated_result["beam_input_ids"]
            logits_guider.interrupt.input_ids = beam_input_ids

        if "selected_beam_indices" in generated_result:
            selected_beam_indices = generated_result["selected_beam_indices"]
            self.track_beam_selections(selected_beam_indices, commentors, code_utils)
        # TODO: Check if this is always zero, as the top beam (the one with eos) might be at the "top" with index 0

        last_token_ids = generated_sequences[0]
        last_token_ids = self.remove_padding(last_token_ids)
        decoded_text = self.decode_tokens_remove_interrupt(
            self.interrupt_input_ids(), last_token_ids
        )
        return decoded_text

    def find_interrupt_type(self, interrupt):
        return [
            i for i in self.interrupts if i.input_id == interrupt.interrupt_token_id
        ][0]

    def edit_generation_text_for_completion(
        self, decoded_text, prompt_util, interrupt, commentor, code_util
    ):
        generated_code = prompt_util.get_whole_code(decoded_text)
        generated_code = commentor.remove_old_comments(generated_code)

        interrupt_type = self.find_interrupt_type(interrupt)
        comment = interrupt_type.create_comment(interrupt.interrupt_context)
        edited_generated_code = commentor.insert_comment(generated_code, comment)
        edited_prompt = prompt_util.format(edited_generated_code)
        return edited_prompt

    def edit_input_ids(self, interrupt, edited_prompt, interrupt_beam_index):
        edited_input_ids = self.tokenizer(
            edited_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        input_ids = interrupt.input_ids
        input_ids, edited_input_ids = self.pad_input_ids(input_ids, edited_input_ids)
        input_ids[interrupt_beam_index] = edited_input_ids
        return input_ids

    def create_lsp_logits_processor(self, lsps, prompt_utils, filenames, expand_size):
        return LspLogitsProcessor(
            self.tokenizer, lsps, prompt_utils, filenames, expand_size, self.disabled
        )

    def create_boundary_logits_processor(self):
        return BoundaryLogitsProcessor(self.tokenizer, [".", "("], self.disabled)
    
    def create_comments_logits_processor(self):
        return CommentsLogitsProcessor(self.tokenizer, self.disabled)

    async def complete(self, code: str, repo_root: str, filename: str = "code.py"):
        with self.device_placement():
            return await self._complete(code, repo_root, filename)

    def fix_config(self, config, prompt):
        if "max_new_tokens" in config:
            prompt_len = len(self.tokenizer(prompt).input_ids)
            config["max_length"] = prompt_len + config["max_new_tokens"]
            del config["max_new_tokens"]
        return config

    async def _complete(self, code: str, repo_root: str, filename: str = "code.py"):
        batch_size = 1
        config = self.generation_config.copy()
        beam_size = config["num_beams"] if "num_beams" in config else 1
        # TODO: allow higher batch size
        lsp = await create_lsp_for_language("python", repo_root)
        prompt_util = Prompt(self.tokenizer, self.message_formatter, code)
        prompt_util.init_completion_prompt()
        prompt = prompt_util.format(code)

        parser = Parser()
        parser.set_language(PY_LANGUAGE)
        # NOTE: They need to be stored per beam, as each beam may have different comments, AND THEY NEED TO BE SHUFFLED ACCORDING TO THE SELECTED BEAMS IN THE BEAMSEARCHSCORER
        code_utils = [PythonCodeUtil() for i in range(batch_size * beam_size)]
        commentors = [Commentor(code_utils[i], parser) for i in range(batch_size * beam_size)]

        start_timestamp = time.time()
        config = self.fix_config(config, prompt)
        logits_guider = self.create_lsp_logits_processor(
            [lsp], [prompt_util], [filename], beam_size
        )
        boundary_logits_processor = self.create_boundary_logits_processor()
        comments_logits_processor = self.create_comments_logits_processor()
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        input_ids, _ = self.expand_input_ids(input_ids, config)
        while True:
            decoded_text = self.resume_generation(
                input_ids,
                batch_size,
                logits_guider,
                boundary_logits_processor,
                comments_logits_processor,
                config,
                commentors,
                code_utils,
            )
            interrupt = logits_guider.interrupt
            if interrupt is None:
                result_code = prompt_util.get_whole_code(decoded_text)
                commentor = commentors[0]
                result_code = commentor.remove_all_comments(result_code)
                return result_code[len(prompt_util.initial_text) :]
            stopped_beam_index = self.index_of_beam_to_edit(interrupt.input_ids)
            commentor = commentors[stopped_beam_index]
            code_util = code_utils[stopped_beam_index]
            edited_prompt = self.edit_generation_text_for_completion(
                decoded_text, prompt_util, interrupt, commentor, code_util
            )
            input_ids = self.edit_input_ids(
                interrupt, edited_prompt, stopped_beam_index
            )
            logits_guider.resume()
            # TODO: move into update config for resumption
            if "max_time" in config:
                current_timestamp = time.time()
                time_for_iteration = current_timestamp - start_timestamp
                config["max_time"] -= time_for_iteration
                start_timestamp = current_timestamp
