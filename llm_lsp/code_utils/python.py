import re
from typing import Optional

from docstring_parser import parse

from llm_lsp.code_utils import CodeUtil


class PythonCodeUtil(CodeUtil):
    @property
    def language_code(self) -> str:
        return "py"

    @property
    def indentation_regex(self) -> re.Pattern:
        return r"\s*"

    def extract_comment(self, code_line: str) -> Optional[str]:
        code_line = code_line.strip()
        if code_line.startswith("#"):
            return code_line[1:].strip()
        return None

    def make_single_line_comment(self, text: str) -> str:
        return "# " + text

    def shorten_docstring(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        docstring = parse(text)
        return f"{docstring.short_description}"
