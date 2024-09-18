import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from unstructured.partition.auto import partition

from _defaults import DEFAULT_CHUNK_STRAT
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
    logger.info(f"Processing {input_file}")
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
    else:
        print("Here")
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
    for dirpath, _, files in os.walk(input):
        for file in files:
            input_file = os.path.join(dirpath, file)
            if folder_tags:
                tag = dirpath.replace(input, "")
                if tag.endswith("/"):
                    tag = tag[:-1]
                if tag.startswith("/"):
                    tag = tag[1:]
                if "/" in tag:
                    tag = tag.split("/")[0]
            else:
                tag = None
            parse(
                input_file,
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
    parser.add_argument("--input", type=str, help="input directory")
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
