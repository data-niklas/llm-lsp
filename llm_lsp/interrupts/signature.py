from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt import Prompt
from llm_lsp.commentor import Comment, Lifetime
from typing import Any

TOKEN_ID = "[SIGNATURE_INTERRUPT]"
SIGNATURE_COMMENT_TYPE = "signature"

class SignatureInterrupt(InterruptType):
    def __init__(self, maximum_documentation_length = 500):
        super().__init__(TOKEN_ID)
        self.maximum_documentation_length = maximum_documentation_length

    def create_comment(self, signature_help: Any) -> Comment:
        active_signature = signature_help.signatures[signature_help.active_signature]
        documentation = active_signature.documentation.value
        
        comment = "Signature note: " + active_signature.label.strip()
        if len(documentation) > 0 and len(documentation) < self.maximum_documentation_length:
            comment += (
                '\nSignature note: Documentation is: """'
                + documentation.replace("\n", "\nSignature note: ")
                + '"""'
            )
        return Comment(lifetime=Lifetime.EPHEMERAL, comment=comment, interrupt=SIGNATURE_COMMENT_TYPE)
