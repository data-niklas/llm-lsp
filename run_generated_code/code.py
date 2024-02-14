from tsv.helper import Parser

# 1. Create an instance of Parser and store it in the variable parser
# 2. read and parse the file `data.tsv`
# 3. store the result in the variable `result`
# Note: The file `data.tsv` contains the columns with the fields of data types: (int, str, str, str, bool, datetime)

# TODO: no signature!?
parser = Parser()
result = parser.parse_file("data.tsv")

# Signature note: print(*values: object, sep: Optional[str]=..., end: Optional[str]=..., file: Optional[SupportsWrite[str]]=..., flush: bool=...) -> None
# Signature note: Documentation is: """print(value, ..., sep=' ', end='\\n', file=sys.stdout, flush=False)
# Signature note: 
# Signature note: Prints the values to a stream, or to sys.stdout by default.
# Signature note: Optional keyword arguments:
# Signature note: file:  a file-like object (stream); defaults to the current sys.stdout.
# Signature note: sep:   string inserted between values, default a space.
# Signature note: end:   string appended after the last value, default a newline.
# Signature note: flush: whether to forcibly flush the stream."""
print(result)