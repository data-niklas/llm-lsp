from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt import Prompt
from llm_lsp.commentor import Comment, Lifetime, InsertedComment
from tree_sitter import Tree
from typing import Any
from llm_lsp.code_utils import CodeUtil


TOKEN_ID = "[SIGNATURE_INTERRUPT]"
SIGNATURE_COMMENT_TYPE = "signature"


class SignatureInterrupt(InterruptType):
    def __init__(self, maximum_documentation_length=500):
        super().__init__(TOKEN_ID)
        self.maximum_documentation_length = maximum_documentation_length

    def create_comment(self, signature_help: Any, code_util: CodeUtil) -> Comment:
        active_signature = signature_help.signatures[signature_help.active_signature]
        documentation = active_signature.documentation.value

        comment = "Signature note: " + active_signature.label.strip()
        if len(documentation) >= self.maximum_documentation_length:
            documentation = documentation.split("\n\n")[0][
                : self.maximum_documentation_length
            ]

        documentation = code_util.shorten_docstring(documentation)
        if len(documentation) > 0:
            comment += (
                '\nSignature note: Documentation is: """'
                + documentation.replace("\n", "\nSignature note: ")
                + '"""'
            )

        return Comment(comment=comment, interrupt=SIGNATURE_COMMENT_TYPE, context=signature_help)
