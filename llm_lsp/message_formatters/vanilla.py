from llm_lsp.message_formatters import MessageFormatter
from typing import List, Dict, Any


class VanillaMessageFormatter(MessageFormatter):
    def create_completion_messages(
        self, initial_code: str, system_prompt_enabled: bool
    ) -> List[Dict[str, Any]]:
        system_prompt = "Return only the completion of the given function. Generate readable and simple code."
        user_prompt = f"Complete the following Python function. Return only code."

        if system_prompt_enabled:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [{"role": "user", "content": user_prompt}]
