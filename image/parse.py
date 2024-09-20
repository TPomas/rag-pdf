import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from unstructured.partition.auto import partition

from _defaults import DEFAULT_CHUNK_STRAT
from _utils import get_tag_from_dir
from rag_schema import DataElement, DataType, Document, Metadata


def elements_to_rag_schema(elements: list, tag=None) -> list[DataElement]:
    output_list = Document()
    for element in elements:
        el = element.to_dict()
        if "url" in el["metadata"]:
            source = el["metadata"]["url"]
        elif "filename" in el["metadata"]:
            source = el["metadata"]["filename"]
        else:
            source = "Unknown"
        el["metadata"]["source"] = source
        page_number = el["metadata"].get("page_number", 1)
        url = el["metadata"].get("url", None)
        text_as_html = el["metadata"].get("text_as_html", None)
        output_list.append(
            DataElement(
                id=el["element_id"],
                data_type=DataType(el["type"]),
                content=el["text"],
                metadata=Metadata(
                    source=el["metadata"]["source"],
                    page_number=page_number,
                    url=url,
                    text_as_html=text_as_html,
                    tag=tag,
                ),
            )
        )
    return output_list


def parse(
    input_file,
    output,
    strategy,
    chunking_strategy: str,
    combine_text_under_n_chars: Optional[int] = None,
    max_characters: Optional[int] = None,
    new_after_n_chars: Optional[int] = None,
    tag=None,
) -> None:
    logger.info(f"Processing {input_file}. Using {tag=}")
    elements = partition(
        filename=input_file,
        skip_infer_table_types=[],
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        combine_text_under_n_chars=combine_text_under_n_chars,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
    )
    logger.info("Done parsing elements")
    output_list = elements_to_rag_schema(elements, tag=tag)
    input_file_path = Path(input_file)

    print(f"{input_file_path=} {input_file_path.parent=}, {input_file_path.stem + '.json'=}")
    output_path = Path(os.path.join(output, input_file_path.stem + ".json"))
    if not output_path.parent.exists():
        print(f"Creating {input_file_path.parent.absolute()=}")
        os.makedirs(output_path.parent.absolute())
    with open(output_path, "w") as f:
        logger.info(f"Writing output to {output_path}")
        json.dump(output_list, f, indent=4)


def main(
    input: str,
    output: str,
    strategy: str,
    chunking_strategy: str,
    combine_text_under_n_chars: int,
    max_characters: int,
    new_after_n_chars: int,
    folder_tags: bool = False,
) -> None:
    # The expectation is that input is a directory which contains various subdirs and no files.
    # Each subdir should itself only contain files, and not additional subdirs, and the contents of
    # each subdir will be parsed and tagged together.
    input_path = Path(input)
    if input_path.is_file():
        raise ValueError("Input must be a directory, not a file.")
    for subdir in input_path.iterdir():
        if subdir.is_file():
            raise ValueError(
                f"Input dir is expected to contain only subdirectories and no files. Found file {subdir=}"
            )
        tag = get_tag_from_dir(subdir) if folder_tags else None
        for file in subdir.iterdir():
            if file.is_dir():
                raise ValueError(
                    f"Subdirectories may only contain files and no additional subdirs. Found subdir {file=}"
                )
            parse(
                file,
                output,
                strategy,
                chunking_strategy,
                combine_text_under_n_chars,
                max_characters,
                new_after_n_chars,
                tag,
            )


if __name__ == "__main__":
    print("\n**********  PARSING **********\n")
    parser = argparse.ArgumentParser(description="File Parser")
    parser.add_argument(
        "--input", type=str, help="Input directory containing to-be-parsed subdirectories."
    )
    parser.add_argument("--output", default="./output", help="output directory")
    parser.add_argument("--strategy", default="auto", help="parsing strategy")
    parser.add_argument(
        "--chunking_strategy", default=DEFAULT_CHUNK_STRAT, help="chunking strategy"
    )
    parser.add_argument(
        "--combine_text_under_n_chars", default=None, type=int, help="unstructured setting"
    )
    parser.add_argument("--max_characters", default=None, type=int, help="unstructured setting")
    parser.add_argument("--new_after_n_chars", default=None, type=int, help="unstructured setting")
    parser.add_argument(
        "--folder_tags",
        action="store_true",
        help="folder tags",
    )
    args = parser.parse_args()

    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    logger.info("Starting processing")
    if args.folder_tags:
        logger.info("Using folder names as tags")
    main(**vars(args))
