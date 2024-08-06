import time
from copy import deepcopy

# from llm_lsp.lsp_client import LspClient
from typing import Any, Dict, List

from transformers import AutoTokenizer, GenerationMixin, LogitsProcessorList

from llm_lsp.code_utils.python import PythonCodeUtil
from llm_lsp.config import LspGenerationConfig
from llm_lsp.constants import INTERRUPT_TOKEN, PAD_TOKEN
from llm_lsp.generation_utils.beam_tracking import BeamTracker
from llm_lsp.interrupts import InterruptStoppingCriteria, InterruptType
from llm_lsp.interrupts.completion import CompletionInterrupt
from llm_lsp.interrupts.deprecation import DeprecationInterrupt
from llm_lsp.interrupts.signature import SignatureInterrupt
from llm_lsp.lsp import create_lsp_for_language, language_from_extension
from llm_lsp.lsp.boundary_logits_processor import BoundaryLogitsProcessor
from llm_lsp.lsp.comments_processor import CommentsLogitsProcessor
from llm_lsp.lsp.lsp_processor import LspLogitsProcessor
from llm_lsp.mixins.interrupt_mixin import InterruptMixin
from llm_lsp.mixins.pipeline_mixin import PipelineMixin
from llm_lsp.mixins.token_sequence_edit_mixin import TokenSequenceEditMixin
from llm_lsp.prompt_formatters import PromptFormatter
from llm_lsp.prompt_formatters.default import DefaultPromptFormatter
from llm_lsp.prompt_formatters.vanilla import VanillaPromptFormatter
from llm_lsp.prompt_state import PromptState
from llm_lsp.mixins.log_mixin import LogMixin

DEFAULT_INTERRUPTS = [
    DeprecationInterrupt(),
    SignatureInterrupt(),
    CompletionInterrupt(),
]


class Generator(InterruptMixin, PipelineMixin, TokenSequenceEditMixin, LogMixin):
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: AutoTokenizer,
        generation_config: Dict[str, Any],
        interrupts: List[InterruptType] = DEFAULT_INTERRUPTS,
        config: LspGenerationConfig = LspGenerationConfig(),
    ):
        self.device = model.device
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.prompt_formatter = (
            VanillaPromptFormatter()
            if config.is_disabled()
            else DefaultPromptFormatter()
        )
        self.interrupts = interrupts
        self.config = config
        self.beam_tracker = BeamTracker()
        self.add_special_tokens()

    def add_special_tokens(self):
        additional_special_tokens = [INTERRUPT_TOKEN]
        if self.tokenizer.pad_token_id is None:
            additional_special_tokens.append(PAD_TOKEN)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                PAD_TOKEN
            )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def decode_tokens_remove_interrupt(self, interrupt_token_id, output_ids):
        if output_ids[-1] == interrupt_token_id:
            tokens = output_ids[:-1]
            return self.tokenizer.decode(tokens, skip_special_tokens=False)
        # Remove eos token
        output_ids = output_ids[:-1]
        if output_ids[-1] == interrupt_token_id:
            return self.tokenizer.decode(output_ids[:-1], skip_special_tokens=False)
        return self.tokenizer.decode(output_ids, skip_special_tokens=False)

    def track_beam_selections(self, prompt_states, code_utils):
        """Beam indices for each batch. Access the selected beams using batch_index * batch_size + beam_index_in_batch

        Args:
            beam_indices (_type_): _description_
        """
        # TODO: add batching
        new_prompt_states = []
        new_codeutils = []
        indices = self.beam_tracker.get_final_beam_indices()
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
        lsp_processor.resume()
        logits_processors = [lsp_processor]
        if self.config.boundary_processor:
            logits_processors.append(boundary_logits_processor)
        if self.config.comments_processor:
            logits_processors.append(comments_logits_processor)
        kwargs = {}
        if "pad_token_id" not in config:
            kwargs["pad_token_id"] = (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        if self.beam_tracker.is_beam_search():
            # Needed because logits processor is called before the process method in which the beam arrangements are tracked
            self.beam_tracker.reset()

        generated_result = self.resume(
            input_ids,
            batch_size,
            logits_processor=LogitsProcessorList(logits_processors),
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

    def create_lsp_logits_processor(self, lsps, prompt_states, file_names, expand_size):
        return LspLogitsProcessor(
            self.tokenizer,
            lsps,
            prompt_states,
            file_names,
            self.interrupt_token_id(),
            expand_size,
            self.beam_tracker,
            self.config,
        )

    def create_boundary_logits_processor(self):
        return BoundaryLogitsProcessor(self.tokenizer, [".", "("])

    def create_comments_logits_processor(self):
        return CommentsLogitsProcessor(self.tokenizer)

    def fix_config(self, config, prompt):
        if "max_new_tokens" in config:
            prompt_len = len(self.tokenizer(prompt).input_ids)
            config["max_length"] = prompt_len + config["max_new_tokens"]
            del config["max_new_tokens"]
        if "max_interrupts" in config:
            del config["max_interrupts"]
        return config

    def initialize_generation_config(self):
        config = self.generation_config.copy()
        max_interrupts = (
            config["max_interrupts"] if "max_interrupts" in config else None
        )
        beam_size = config["num_beams"] if "num_beams" in config else 1
        return config, max_interrupts, beam_size

    async def initialize_generation_state(self, code: str, repo_root: str, file_name: str):
        config, max_interrupts, beam_size = self.initialize_generation_config()
        batch_size = 1
        boundary_logits_processor = self.create_boundary_logits_processor()
        comments_logits_processor = self.create_comments_logits_processor()
        language = language_from_extension(file_name)
        if language is None:
            raise "The language is not supported"
        lsp = await create_lsp_for_language(language, repo_root)
        prompt_states = [
            PromptState(self.tokenizer, self.prompt_formatter, code)
            for i in range(batch_size * beam_size)
        ]

        # NOTE: They need to be stored per beam, as each beam may have different comments, AND THEY NEED TO BE SHUFFLED ACCORDING TO THE SELECTED BEAMS IN THE BEAMSEARCHSCORER
        code_utils = [PythonCodeUtil() for i in range(batch_size * beam_size)]
        lsp_processor = self.create_lsp_logits_processor(
            [lsp], prompt_states, [file_name], beam_size
        )
        return (
            config,
            max_interrupts,
            beam_size,
            batch_size,
            boundary_logits_processor,
            comments_logits_processor,
            lsp_processor,
            prompt_states,
            code_utils,
        )

    def retrieve_final_code(self, prompt_state: PromptState, decoded_text: str) -> str:
        result_code = prompt_state.get_whole_code(decoded_text)
        self.log_code(result_code, "END")
        return result_code[len(prompt_state.initial_text) :]

    async def _complete(self, code: str, repo_root: str, file_name: str = "code.py"):
        self.log_code(code, "START")
        interrupt_count = 0
        # TODO: allow higher batch size
        (
            config,
            max_interrupts,
            beam_size,
            batch_size,
            boundary_logits_processor,
            comments_logits_processor,
            lsp_processor,
            prompt_states,
            code_utils,
        ) = await self.initialize_generation_state(code, repo_root, file_name)


        start_timestamp = time.time()
        prompt = prompt_states[0].format(code)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        config = self.fix_config(config, prompt)
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
                return self.retrieve_final_code(prompt_state, decoded_text)
            stopped_beam_index = self.index_of_beam_to_edit(interrupt.input_ids)
            prompt_state = prompt_states[stopped_beam_index]
            if max_interrupts is not None:
                if max_interrupts == 0:
                    return self.retrieve_final_code(prompt_state, decoded_text)
                max_interrupts = max_interrupts - 1
            code_util = code_utils[stopped_beam_index]
            edited_prompt = self.edit_generation_text_for_completion(
                decoded_text, prompt_state, interrupt, code_util
            )
            input_ids = self.edit_input_ids(
                interrupt, edited_prompt, stopped_beam_index
            )
            # TODO: move into update config for resumption
            interrupt_count += 0
            if "max_time" in config:
                current_timestamp = time.time()
                time_for_iteration = current_timestamp - start_timestamp
                config["max_time"] -= time_for_iteration
                start_timestamp = current_timestamp
                if config["max_time"] < 0:
                    return self.retrieve_final_code(prompt_state, decoded_text)
            self.log_interruption(input_ids, interrupt_count)

    async def complete(self, code: str, repo_root: str, file_name: str = "code.py"):
        with self.device_placement():
            return await self._complete(code, repo_root, file_name)
