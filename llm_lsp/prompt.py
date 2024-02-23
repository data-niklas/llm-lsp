class Prompt:
    def __init__(self, template, tokenizer):
        self.tokenizer = tokenizer
        self.template = template

    def wrap_in_code_block(self, code):
        # TODO: generalize the py
        return "```py\n" + code

    def format(self, code, new_code) -> str:
        messages = self.template(code)
        self.code = code
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        self.prompt = prompt
        # new_code is not part of the prompt, but part of the output
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
        generated_text = text[len(self.prompt):]
        generated_code = self.extract_from_markdown_code_block(generated_text)
        return generated_code