from jinja2 import TemplateError
from typing import List, Dict, Any
from transformers import AutoTokenizer
from llm_lsp.message_formatters import MessageFormatter

class Prompt:
    def __init__(self, tokenizer: AutoTokenizer, message_formatter: MessageFormatter, initial_text: str):
        self.tokenizer = tokenizer
        self.initial_text = initial_text
        self.message_formatter = message_formatter
        self.instructions = []

    def init_completion_prompt(self):
        """Create a complete prompt with special tokens and ready to use for generation"""
        try:
            messages = self.message_formatter.create_completion_messages(self.initial_text, True)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # new_code is not part of the prompt, but part of the output
        except TemplateError:
            messages = self.message_formatter.create_completion_messages(self.initial_text, False)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.initial_prompt = prompt
        self.code_prefix = self.initial_text

    def init_prompt_with_instructions(self):
        """Create a complete prompt with special tokens and ready to use for generation"""
        i = " ".join([c.comment + "." for c in self.instructions])
        try:
            messages = self.message_formatter.create_completion_messages(self.initial_text, True)
            for message in messages:
                if message["role"] == "user":
                    message["content"] += ". " + i
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # new_code is not part of the prompt, but part of the output
        except TemplateError:
            messages = self.message_formatter.create_completion_messages(self.initial_text, False)
            for message in messages:
                if message["role"] == "user":
                    message["content"] += ". " + i
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.initial_prompt = prompt
        self.code_prefix = self.initial_text

    def add_comment(self, comment, name):
        self.instructions = [i for i in self.instructions if i.interrupt != name]
        if comment is not None:
            self.instructions.append(comment)
        self.init_prompt_with_instructions() 

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
