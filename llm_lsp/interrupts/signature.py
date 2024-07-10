from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt_state import Comment
from tree_sitter import Tree
from typing import Any, Optional
from llm_lsp.code_utils import CodeUtil
import re

SIGNATURE_COMMENT_TYPE = "signature"

def has_urls(string):
 
    # findall() has been used
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return len(url) > 0
 

class SignatureInterrupt(InterruptType):
    def __init__(self, maximum_documentation_length=500):
        super().__init__()
        self.maximum_documentation_length = maximum_documentation_length

    def type_name(self) -> str:
        return SIGNATURE_COMMENT_TYPE

    def create_comment(self, signature_help: Any, code_util: CodeUtil) -> Optional[Comment]:
        if len(signature_help.signatures) == 0:
            return None
        active_signature = signature_help.signatures[signature_help.active_signature]
        documentation = active_signature.documentation.value

        comment = "Hint: The code item has the following signature: " + active_signature.label.strip()

        documentation = code_util.shorten_docstring(documentation)
        # Do not add documentation linking to web pages:
        if len(documentation) > 0 and not has_urls(documentation):
            comment += "\nThe code item has the following documentation:\n\t" + documentation.replace("\n", "\n\t")
        #comment += "\nOnly provide necessary arguments. Provide positional arguments without name."# and omit default argument values."
        #comment += "\nFirst provide necessary positional arguments. Then provide necessary named arguments with non-default values. Often times the default values of named arguments are sufficient and may be omitted."
        comment += "\nProvide as few arguments as possible and as many as needed. Provide arguments as positional arguments instead of as named arguments if possible. Often times optional arguments can be omitted"
        return Comment(comment=comment, interrupt=SIGNATURE_COMMENT_TYPE, context=signature_help)