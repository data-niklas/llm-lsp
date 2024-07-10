# from llm_lsp.lsp_client import LspClient


from llm_lsp.code_utils import CodeUtil
from llm_lsp.interrupts import Interrupt
from llm_lsp.interrupts.completion import COMPLETION_COMMENT_TYPE
from llm_lsp.interrupts.deprecation import DEPRECATION_COMMENT_TYPE
from llm_lsp.mixins.padding_mixin import PaddingMixin
from llm_lsp.prompt_formatters import PromptFormatter


class TokenSequenceEditMixin(PaddingMixin):
    def edit_generation_text_for_completion(
        self, decoded_text, prompt_state, interrupt, code_util
    ):
        generated_code = prompt_state.get_whole_code(decoded_text)
        interrupt_type = self.find_interrupt_type(interrupt)
        if interrupt_type.type_name() == COMPLETION_COMMENT_TYPE and self.config.predict_correct_completion_symbol:
            dep_interrupt_type = [
                i for i in self.interrupts if i.type_name() == DEPRECATION_COMMENT_TYPE
            ][0]
            comment = dep_interrupt_type.create_comment(
                interrupt.interrupt_context["deprecation"], code_util
            )
            generated_code += self.determine_next_symbol_from_completions(
                interrupt.interrupt_context["completion"],
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
        self.log_code(prompt, "PREDICT CORRECT COMPLETION SYMBOL START")
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        generation_result = self.model.generate(
            input_ids, use_cache=True, **self.generation_config
        )
        generation_text = self.tokenizer.decode(
            generation_result[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        predicted_completion_symbol = ""
        if generation_text.endswith("`"):
            predicted_completion_symbol = generation_text.split("`")[0]
        self.log_code(predicted_completion_symbol, "PREDICT CORRECT COMPLETION SYMBOL END")
        return predicted_completion_symbol
