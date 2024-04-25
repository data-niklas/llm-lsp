import re

def determine_indentation(line: str) -> str:
    non_whitespace_match = re.search(r"\S", line)
    if non_whitespace_match is None:
        return ""
    non_whitespace_index = non_whitespace_match.start()
    return line[:non_whitespace_index]