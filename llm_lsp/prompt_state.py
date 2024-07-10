from jinja2 import TemplateError
from typing import List, Dict, Any
from transformers import AutoTokenizer
from llm_lsp.prompt_formatters import PromptFormatter
from dataclasses import dataclass

@dataclass
class Comment:
    comment: str
    interrupt: str
    context: Any

class PromptState:
    def __init__(self, tokenizer: AutoTokenizer, prompt_formatter: PromptFormatter, initial_text: str):
        self.tokenizer = tokenizer
        self.initial_text = initial_text
        self.prompt_formatter = prompt_formatter
        self.instructions = []
        self._create_completion_prompt()

    def _create_completion_prompt(self):
        """Create a complete prompt with special tokens and ready to use for generation"""
        i = "\n".join([c.comment + "." for c in self.instructions])
        try:
            messages = self.prompt_formatter.create_completion_messages(self.initial_text, True, i)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # new_code is not part of the prompt, but part of the output
        except TemplateError:
            messages = self.prompt_formatter.create_completion_messages(self.initial_text, False, i)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.initial_prompt = prompt
        self.code_prefix = self.initial_text


    def add_comment(self, comment, name):
        self.instructions = [i for i in self.instructions if i.interrupt != name]
        if comment is not None:
            self.instructions.append(comment)
        self._create_completion_prompt() 

    def get_comment_of_interrupt(self, name):
        results = [i for i in self.instructions if i.interrupt == name]
        if len(results) == 0:
            return None
        return results[0]

    def wrap_in_code_block(self, code):
        # TODO: generalize the py
        return "```py\n" + code

    def format(self, new_code) -> str:
        return self.initial_prompt + self.wrap_in_code_block(new_code)

    def extract_from_markdown_code_block(self, text: str):
        lines = text.splitlines()
        if len(lines) > 0 and "```" in lines[0]:
            lines = lines[1:]
            text = "\n".join(lines)
            return text.split("```")[0].strip()
        return "\n".join(lines)

    def get_naughty_tokens(self):
        return self.tokenizer.added_tokens_encoder.keys()

    def get_generated_code(self, text: str):
        # Remove trailing " " after special tokens in some tokenizers to actually make encoding tokens reversible
        for naughty_token in self.get_naughty_tokens():
            text = text.replace(naughty_token + " ", naughty_token)
        generated_text = text[len(self.initial_prompt) :]
        generated_code = self.extract_from_markdown_code_block(generated_text)
        return generated_code[len(self.code_prefix):]

    def get_whole_code(self, text):
        generated_code = self.get_generated_code(text)
        return self.code_prefix + generated_code
