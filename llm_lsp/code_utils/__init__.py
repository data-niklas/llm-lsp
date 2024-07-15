import re
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional


class CodeUtil(ABC):
    @abstractproperty
    def language_code(self) -> str:
        pass

    @abstractproperty
    def dereference_operator(self) -> str:
        pass

    @abstractproperty
    def indentation_regex(self) -> re.Pattern:
        pass

    @abstractmethod
    def extract_comment(self, code_line: str) -> Optional[str]:
        pass

    @abstractmethod
    def make_single_line_comment(self, text: str) -> str:
        pass

    def wrap_in_start_of_code_block(self, code: str) -> str:
        return f"```{self.language_code}\n" + code

    def wrap_in_code_block(self, code: str) -> str:
        return self.wrap_in_start_of_code_block(code) + "\n```"

    def get_indentation_prefix(self, code_line: str) -> str:
        indentation_match = re.search(self.indentation_regex, code_line)
        if indentation_match is None:
            return ""
        indentation_end = indentation_match.end()
        return code_line[:indentation_end]

    @abstractmethod
    def shorten_docstring(self, text: str) -> str:
        pass
