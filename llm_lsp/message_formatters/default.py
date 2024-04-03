from llm_lsp.message_formatters import MessageFormatter
from typing import List, Dict, Any


class DefaultMessageFormatter(MessageFormatter):
    def create_completion_messages(
        self, initial_code: str, system_prompt_enabled: bool
    ) -> List[Dict[str, Any]]:
        system_prompt = "Return only the completion of the given function. Follow the instructions in the code comments for additional instructions on how to complete the code. Generate readable and simple code."
        user_prompt = f"Complete the following Python function. Return only code."
        if system_prompt_enabled:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [{"role": "user", "content": user_prompt}]

    def create_generation_messages(
        self, initial_user_instruction: str, system_prompt_enabled: bool
    ) -> List[Dict[str, Any]]:
        system_prompt = "Follow the instructions in the code comments for additional instructions on how to generate the code. Return only the generation."
        if system_prompt_enabled:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_user_instruction},
            ]
        return [{"role": "user", "content": initial_user_instruction}]
