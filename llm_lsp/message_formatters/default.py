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


    def create_completion_chooser_message(self, wrapped_current_code: str, system_prompt_enabled: bool, completions, deprecation) -> List[Dict[str, Any]]:
        completion_text = ", ".join(["`" + item.insert_text + "`" for item in completions])
        deprecation_text = "" if deprecation is None else deprecation.comment
        user_prompt = "The following symbols are code completion entries. Determine the appropriate symbol to complete the code in the code block: " + completion_text + "\n\n" + wrapped_current_code + "\n\nReturn only the single chosen symbol. Do not provide commentary. Output format:\n`CHOSEN SYMBOL`\n\n" + deprecation_text
        return [{"role": "user", "content": user_prompt}]