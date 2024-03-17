from llm_lsp.interrupts import Interrupt
from llm_lsp.prompt import Prompt
from llm_lsp.code_utils import determine_indentation
from typing import Any

class SignatureInterrupt(Interrupt):
    def __init__(self, maximum_documentation_length = 500):
        super().__init__("[SIGNATURE_INTERRUPT]")
        self.maximum_documentation_length = maximum_documentation_length

    def add_signature_notes(self, code: str, signature_help):
        try:
            first_lines, last_line = code.rsplit("\n", 1)
        except ValueError:
            first_lines, last_line = "", code
        indentation = determine_indentation(last_line)

        active_signature = signature_help.signatures[signature_help.active_signature]
        documentation = active_signature.documentation.value
        comments_text = indentation + "# Signature note: " + active_signature.label.strip()
        if len(documentation) > 0 and len(documentation) < self.maximum_documentation_length:
            comments_text += (
                "\n"
                + indentation
                + '# Signature note: Documentation is: """'
                + documentation.replace("\n", "\n" + indentation + "# Signature note: ")
                + '"""'
            )
        return first_lines + "\n" + comments_text + "\n" + last_line


    def edit_generated_code_for_completion(self, generated_code: str, context: Any) -> str:
        generated_code_with_notes = self.add_signature_notes(generated_code, context)
        return generated_code_with_notes