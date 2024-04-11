from llm_lsp.code_utils import CodeUtil
import re
from typing import Optional

class PythonCodeUtil(CodeUtil):
    @property
    def language_code(self) -> str:
        return "py"

    @property
    def indentation_regex(self) -> re.Pattern:
        return r"\s*"

    def extract_comment(self, code_line: str) -> Optional[str]:
        code_line = code_line.strip()
        if line.startswith("#"):
            return line[1:].strip()
        return None

    def make_single_line_comment(self, text: str) -> str:
        return "# " + text