class LogMixin:
    def log(self, text: str):
        with open(self.config.chat_history_log_file, "a") as f:
            f.write(text)

    def log_interruption(self, input_ids, interrupt_index):
        if self.config.chat_history_log_file is None:
            return
        code_snippets = self.tokenizer.batch_decode(input_ids)
        text = f"INTERRUPT {interrupt_index}\n"
        for i, code_snippet in enumerate(code_snippets):
            text += f"BEAM {i}:\n```\n{code_snippet}\n```\n"
        text += "\n"
        self.log(text)

    def log_code(self, code: str, point: str):
        if self.config.chat_history_log_file is None:
            return
        text = f"{point}:\n```\n{code}\n```\n\n"
        self.log(text)