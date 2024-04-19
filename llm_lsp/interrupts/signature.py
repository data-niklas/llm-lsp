from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt import Prompt
from llm_lsp.commentor import Comment, Lifetime, InsertedComment
from tree_sitter import Tree
from typing import Any
from llm_lsp.generator import PY_LANGUAGE
from llm_lsp.code_utils import CodeUtil

PY_PAREN_OPEN_QUERY = PY_LANGUAGE.query("(\"(\") @element")

TOKEN_ID = "[SIGNATURE_INTERRUPT]"
SIGNATURE_COMMENT_TYPE = "signature"

class SignatureInterrupt(InterruptType):
    def __init__(self, maximum_documentation_length = 500):
        super().__init__(TOKEN_ID)
        self.maximum_documentation_length = maximum_documentation_length

    def is_comment_old(code: str, tree: Tree, comment: InsertedComment) -> bool:
        captures = PY_PAREN_OPEN_QUERY.captures(tree.root_node)
        all_parens = [capture[0] for capture in captures]
        parens_after_comment = [paren for paren in all_parens if paren.start_point[0] >= comment.end_line]
        if len(parens_after_comment) == 0:
            return False
        last_paren = parens_after_comment[-1]
        # TODO: save node in comment instead of finding it
        last_sibling = last_paren.parent.children[-1]
        return last_sibling.type == ")"

    def create_comment(self, signature_help: Any, code_util: CodeUtil) -> Comment:
        active_signature = signature_help.signatures[signature_help.active_signature]
        documentation = active_signature.documentation.value
        
        comment = "Signature note: " + active_signature.label.strip()
        if len(documentation) >= self.maximum_documentation_length:
            documentation = documentation.split("\n\n")[0][:self.maximum_documentation_length]

        documentation = code_util.shorten_docstring(documentation)
        if len(documentation) > 0:
            comment += (
                '\nSignature note: Documentation is: """'
                + documentation.replace("\n", "\nSignature note: ")
                + '"""'
            )

        return Comment(is_old=SignatureInterrupt.is_comment_old, comment=comment, interrupt=SIGNATURE_COMMENT_TYPE)
