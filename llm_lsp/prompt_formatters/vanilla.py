from typing import Any, Dict, List

from llm_lsp.prompt_formatters import PromptFormatter


class VanillaPromptFormatter(PromptFormatter):
    def create_completion_messages(
        self, initial_code: str, system_prompt_enabled: bool, instruction_text: str
    ) -> List[Dict[str, Any]]:
        system_prompt = "Return only the completion of the given function. Generate readable and simple code."
        user_prompt = "Complete the following Python function. Return only code."

        if system_prompt_enabled:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [{"role": "user", "content": user_prompt}]

    def create_completion_chooser_message(
        self, wrapped_current_code: str, system_prompt_enabled: bool, completions
    ) -> List[Dict[str, Any]]:
        pass
