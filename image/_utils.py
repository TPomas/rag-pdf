from pathlib import Path
from typing import Union


def get_tag_from_dir(input_dir: Union[str, Path]) -> str:
    input_path = Path(input_dir)
    if input_path.is_file():
        raise ValueError("input expected to be a directory, not a file")
    tag = input_path.parts[-1]
    return tag
