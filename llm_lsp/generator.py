from transformers import AutoTokenizer, AutoModel, PreTrainedModel, GenerationMixin
from transformers.generation import GenerationMode, GenerationConfig
import warnings
from tree_sitter import Language, Parser
import tree_sitter_python

PY_LANGUAGE = Language(tree_sitter_python.language(), "python")
# from llm_lsp.lsp_client import LspClient
from typing import Dict, Any, Optional, List
from llm_lsp.prompt_state import PromptState
from llm_lsp.prompt_formatters import PromptFormatter
from llm_lsp.prompt_formatters.default import DefaultPromptFormatter
from llm_lsp.prompt_formatters.vanilla import VanillaPromptFormatter
from pygls.lsp.client import BaseLanguageClient
from llm_lsp.interrupts import InterruptType, InterruptStoppingCriteria, Interrupt
from llm_lsp.interrupts.deprecation import DeprecationInterrupt, DEPRECATION_COMMENT_TYPE
from llm_lsp.interrupts.signature import SignatureInterrupt, SIGNATURE_COMMENT_TYPE
from llm_lsp.interrupts.completion import (
    CompletionInterrupt, COMPLETION_COMMENT_TYPE
)
from llm_lsp.lsp import create_lsp_for_language
from llm_lsp.interrupt_mixin import resume
from llm_lsp.lsp.lsp_processor import LspLogitsProcessor
from llm_lsp.lsp.boundary_logits_processor import BoundaryLogitsProcessor
from llm_lsp.lsp.comments_processor import CommentsLogitsProcessor
from llm_lsp.code_utils import CodeUtil
from llm_lsp.code_utils.python import PythonCodeUtil
from llm_lsp.generation_utils.beam_tracking import BeamTracker
from llm_lsp.constants import INTERRUPT_TOKEN
import torch.nn.functional as F
import os
import torch
from contextlib import contextmanager
from copy import deepcopy
import copy
import time

DEFAULT_INTERRUPTS = [
    DeprecationInterrupt(),
    SignatureInterrupt(),
    CompletionInterrupt(),
]

PAD_TOKEN = "[PAD]"


class Generator:
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: AutoTokenizer,
        generation_config: Dict[str, Any],
        prompt_formatter: PromptFormatter = None,
        interrupts: List[InterruptType] = DEFAULT_INTERRUPTS,
        disabled=False,
    ):
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        if prompt_formatter is None:
            prompt_formatter = (
                VanillaPromptFormatter() if disabled else DefaultPromptFormatter()
            )
        self.prompt_formatter = prompt_formatter
        self.interrupts = interrupts
        self.disabled = disabled
        self.beam_tracker = BeamTracker()
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
        additional_special_tokens = [INTERRUPT_TOKEN, PAD_TOKEN]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def remove_padding(self, tokens):
        token_id = self.tokenizer.pad_token_id
        index = (tokens != token_id).nonzero()[0].item()
        return tokens[index:]

    def remove_nd_padding(self, tokens):
        token_id = self.tokenizer.pad_token_id
        index = (tokens != token_id).nonzero()[:, -1].min().item()
        return tokens[:, index:]

    def decode_tokens_remove_interrupt(self, interrupt_token_id, output_ids):
        if output_ids[-1] == interrupt_token_id:
            tokens = output_ids[:-1]
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        # Remove eos token
        output_ids = output_ids[:-1]
        if output_ids[-1] == interrupt_token_id:
            return self.tokenizer.decode(output_ids[:-1], skip_special_tokens=False)
        return self.tokenizer.decode(output_ids, skip_special_tokens=False)

    def pad_input_ids(self, input_ids, edited_input_id):
        pad_token_id = self.tokenizer.pad_token_id
        edited_len = edited_input_id.shape[1]
        inputs_len = input_ids.shape[1]
        pad_to_len = max(edited_len, inputs_len)
        edited_input_id = F.pad(
            edited_input_id, (pad_to_len - edited_len, 0), value=pad_token_id
        )
        input_ids = F.pad(input_ids, (pad_to_len - inputs_len, 0), value=pad_token_id)
        return input_ids, edited_input_id


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

    def track_beam_selections(self, prompt_states, code_utils):
        """Beam indices for each batch. Access the selected beams using batch_index * batch_size + beam_index_in_batch

        Args:
            beam_indices (_type_): _description_
        """
        # TODO: add batching
        new_prompt_states = []
        new_codeutils = []
        indices = self.beam_tracker.get_final_beam_indices()
        a = None
        for i in range(len(indices)):
            index = indices[i]
            prompt_state = prompt_states[index]
            code_util = code_utils[index]
            prompt_state_copy = deepcopy(prompt_state)
            new_prompt_states.append(prompt_state_copy)
            new_codeutils.append(deepcopy(code_util))

        # Change in place
        for i in range(len(prompt_states)):
            prompt_states[i] = new_prompt_states[i]
            code_utils[i] = new_codeutils[i]
        a = None

    def interrupt_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(INTERRUPT_TOKEN)

    def index_of_beam_to_edit(self, input_ids):
        eos_tokens = [self.interrupt_token_id()]
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
        lsp_processor,
        boundary_logits_processor,
        comments_logits_processor,
        config,
        prompt_states,
        code_utils,
    ):
        """
        Returns the decoded text for each batch. If using beam search this is the decoded text of the best beam. In addition this method will return the index of the best beam. This is necessary to access the appropriate comment tool.
        """
        stopping_criterium = InterruptStoppingCriteria(self.interrupt_token_id())
        logits_processor = [
            lsp_processor,
            boundary_logits_processor,
            comments_logits_processor,
        ]
        # logits_processor = [lsp_processor, comments_logits_processor]
        kwargs = {}
        if "pad_token_id" not in config:
            kwargs["pad_token_id"] = (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        if self.beam_tracker.is_beam_search():
            # Needed because logits processor is called before the process method in which the beam arrangements are tracked
            self.beam_tracker.reset()

        generated_result = resume(
            self.model,
            input_ids,
            batch_size,
            logits_processor=logits_processor,
            stopping_criteria=[stopping_criterium],
            return_dict_in_generate=True,
            output_scores=True,
            beam_tracker=self.beam_tracker,
            use_cache=True,
            **config,
        )
        generated_sequences = generated_result["sequences"]

        if "beam_input_ids" in generated_result and lsp_processor.interrupt is not None:
            beam_input_ids = generated_result["beam_input_ids"]
            # Remove the last token to fix top-k logits warper selecting nonsensical tokens
            # This bug occurs when the interrupted beam selects an interrupt token with a high logits value and top-k samples from all other tokens randomly
            lsp_processor.interrupt.input_ids = beam_input_ids[:, :-1]

        if self.beam_tracker.is_beam_search():
            self.track_beam_selections(prompt_states, code_utils)
        # TODO: Check if this is always zero, as the top beam (the one with eos) might be at the "top" with index 0

        last_token_ids = generated_sequences[0]
        last_token_ids = self.remove_padding(last_token_ids)
        decoded_text = self.decode_tokens_remove_interrupt(
            self.interrupt_token_id(), last_token_ids
        )
        return decoded_text

    def find_interrupt_type(self, interrupt: Interrupt):
        return [
            i for i in self.interrupts if i.type_name() == interrupt.interrupt_type_name
        ][0]

    def determine_next_symbol_from_completions(
        self,
        completions,
        deprecation_text,
        prompt_formatter: PromptFormatter,
        code_util: CodeUtil,
        code,
    ):
        wrapped_code = code_util.wrap_in_code_block(code)
        messages = prompt_formatter.create_completion_chooser_message(
            wrapped_code, False, completions, deprecation_text
        )
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += "`"
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        generation_result = self.model.generate(
            input_ids, use_cache=True, **self.generation_config
        )
        generation_text = self.tokenizer.decode(
            generation_result[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        if generation_text.endswith("`"):
            return generation_text.split("`")[0]
        return ""

    def edit_generation_text_for_completion(
        self, decoded_text, prompt_state, interrupt, code_util
    ):
        generated_code = prompt_state.get_whole_code(decoded_text)
        interrupt_type = self.find_interrupt_type(interrupt)
        if interrupt_type.type_name() == COMPLETION_COMMENT_TYPE:
            dep_interrupt_type = [
                i for i in self.interrupts if i.type_name() == DEPRECATION_COMMENT_TYPE
            ][0]
            comment = dep_interrupt_type.create_comment(
                interrupt.interrupt_context["dep"], code_util
            )
            generated_code += self.determine_next_symbol_from_completions(
                interrupt.interrupt_context["comp"],
                comment,
                prompt_state.prompt_formatter,
                code_util,
                generated_code,
            )
        else:
            comment = interrupt_type.create_comment(
                interrupt.interrupt_context, code_util
            )
            prompt_state.add_comment(comment, interrupt_type.type_name())
        edited_prompt = prompt_state.format(generated_code)
        return edited_prompt

    def edit_input_ids(self, interrupt, edited_prompt, interrupt_beam_index):
        edited_input_ids = self.tokenizer(
            edited_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        input_ids = interrupt.input_ids
        if input_ids.shape[0] > 1:
            input_ids, edited_input_ids = self.pad_input_ids(
                input_ids, edited_input_ids
            )
            input_ids[interrupt_beam_index] = edited_input_ids
        else:
            input_ids = edited_input_ids
        input_ids = self.remove_nd_padding(input_ids)
        return input_ids

    def create_lsp_logits_processor(self, lsps, prompt_states, filenames, expand_size):
        return LspLogitsProcessor(
            self.tokenizer,
            lsps,
            prompt_states,
            filenames,
            self.interrupt_token_id(),
            expand_size,
            self.beam_tracker,
            self.disabled,
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
        max_interrupts = (
            config["max_interrupts"] if "max_interrupts" in config else None
        )
        if max_interrupts is not None:
            del config["max_interrupts"]
        beam_size = config["num_beams"] if "num_beams" in config else 1
        # TODO: allow higher batch size
        lsp = await create_lsp_for_language("python", repo_root)
        prompt_states = [
            PromptState(self.tokenizer, self.prompt_formatter, code)
            for i in range(batch_size * beam_size)
        ]
        prompt = prompt_states[0].format(code)

        parser = Parser()
        parser.set_language(PY_LANGUAGE)
        # NOTE: They need to be stored per beam, as each beam may have different comments, AND THEY NEED TO BE SHUFFLED ACCORDING TO THE SELECTED BEAMS IN THE BEAMSEARCHSCORER
        code_utils = [PythonCodeUtil() for i in range(batch_size * beam_size)]

        start_timestamp = time.time()
        config = self.fix_config(config, prompt)
        lsp_processor = self.create_lsp_logits_processor(
            [lsp], prompt_states, [filename], beam_size
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
                lsp_processor,
                boundary_logits_processor,
                comments_logits_processor,
                config,
                prompt_states,
                code_utils,
            )
            interrupt = lsp_processor.interrupt
            if interrupt is None:
                prompt_state = prompt_states[0]
                result_code = prompt_state.get_whole_code(decoded_text)
                return result_code[len(prompt_state.initial_text) :]
            stopped_beam_index = self.index_of_beam_to_edit(interrupt.input_ids)
            prompt_state = prompt_states[stopped_beam_index]
            if max_interrupts is not None:
                if max_interrupts == 0:
                    result_code = prompt_state.get_whole_code(decoded_text)
                    return result_code[len(prompt_state.initial_text) :]
                max_interrupts = max_interrupts - 1
            code_util = code_utils[stopped_beam_index]
            edited_prompt = self.edit_generation_text_for_completion(
                decoded_text, prompt_state, interrupt, code_util
            )
            input_ids = self.edit_input_ids(
                interrupt, edited_prompt, stopped_beam_index
            )
            lsp_processor.resume()
            # TODO: move into update config for resumption
            if "max_time" in config:
                current_timestamp = time.time()
                time_for_iteration = current_timestamp - start_timestamp
                config["max_time"] -= time_for_iteration
                start_timestamp = current_timestamp
                if config["max_time"] < 0:
                    result_code = prompt_state.get_whole_code(decoded_text)
                    return result_code[len(prompt_state.initial_text) :]
