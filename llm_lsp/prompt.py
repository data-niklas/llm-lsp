from llm_lsp.strategy import GenerationStrategy
from jinja2 import TemplateError

class Prompt:
    def __init__(self, template, tokenizer):
        self.tokenizer = tokenizer
        self.template = template

    def wrap_in_code_block(self, code):
        # TODO: generalize the py
        return "```py\n" + code

    def format(self, code, new_code, strategy) -> str:
        self.code = code
        try:
            messages = self.template(code, strategy, True)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # new_code is not part of the prompt, but part of the output
        except TemplateError:
            messages = self.template(code, strategy, False)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
        self.prompt = prompt
        return prompt + self.wrap_in_code_block(new_code)

    def extract_from_markdown_code_block(self, text: str):
        lines = text.splitlines()
        if len(lines) > 0 and "```" in lines[0]:
            lines = lines[1:]

        if len(lines) > 0 and "```" in lines[-1]:
            lines = lines[:-1]
        return "\n".join(lines)

    def get_generated_code(self, text):
        text = text.replace("<s> ", "<s>")
        generated_text = text[len(self.prompt) :]
        generated_code = self.extract_from_markdown_code_block(generated_text)
        return generated_code


def make_prompt(code, strategy, system_prompt_enabled: bool = True):
    if strategy == GenerationStrategy.COMPLETE:
        system_prompt = "Follow the instructions in the code comments for additional instructions on how to complete the code. Return only the completion."
        user_prompt = f"Complete the provided Python code. Return only the completion:\n```py\n{code}\n```"
        if system_prompt_enabled:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [{"role": "user", "content": system_prompt + " " + user_prompt}]
    elif strategy == GenerationStrategy.GENERATE:
        system_prompt = "Follow the instructions in the code comments for additional instructions on how to generate the code. Return only the generation."
        user_prompt = code
        if system_prompt_enabled:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        return [{"role": "user", "content": system_prompt + " " + user_prompt}]
