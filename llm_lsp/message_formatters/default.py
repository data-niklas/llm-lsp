from llm_lsp.message_formatters import MessageFormatter
from typing import List, Dict, Any


class DefaultMessageFormatter(MessageFormatter):
    def create_completion_messages(
        self, initial_code: str, system_prompt_enabled: bool, instruction_text: str
    ) -> List[Dict[str, Any]]:
        system_prompt = "Return only the completion of the given function. Follow the instructions in the code comments for additional instructions on how to complete the code. Generate readable and simple code."
        user_prompt = "Complete the following Python function. Return only code."
        user_prompt += "" if len(instruction_text) == 0 else "\n" + instruction_text
        if system_prompt_enabled:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [{"role": "user", "content": user_prompt}]
