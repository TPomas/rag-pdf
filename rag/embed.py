import argparse
import json
import os

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from rag._defaults import DEFAULT_HF_EMBED_MODEL


def main(data_path: str, path_to_db: str, embed_model: str, db: chromadb.PersistentClient) -> None:
    collection = db.get_or_create_collection(name="documents", metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = []
    index = VectorStoreIndex(docs, storage_context=storage_context, embed_model=embed_model)
    for dirpath, _, files in os.walk(data_path):
        for file in files:
            input_file = os.path.join(dirpath, file)

            with open(input_file, "r") as f:
                input_text = json.load(f)
                for doc in input_text:
                    # (4.26.2024) Andrew: Dealing with issue when parsing txt or xml, doc can be a string
                    if isinstance(doc, dict):
                        if doc["data_type"] == "Table":
                            text = doc["metadata"]["text_as_html"]
                        else:
                            text = doc["content"]
                        source = doc["metadata"]["source"]
                        if "page_number" in doc["metadata"]:
                            page_number = doc["metadata"]["page_number"]
                        else:
                            page_number = 1
                        if "tag" in doc["metadata"]:
                            tag = doc["metadata"]["tag"]
                        else:
                            tag = ""
                        metadata = {
                            "Source": source,
                            "Page Number": page_number,
                            "Commit": os.environ.get("PACH_JOB_ID", ""),
                            "Tag": tag,
                        }
                        docs.append(TextNode(text=text, metadata=metadata))

    print("Number of chunks: ", len(docs))

    index.insert_nodes(docs, show_progress=True)
    print("Indexing done!")
    index.storage_context.persist(persist_dir=path_to_db)
    print(f"Persisting done! Saved at {path_to_db}")


if __name__ == "__main__":
    print("\n**********  EMBEDDING **********\n")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-db",
        type=str,
        default="db/",
        help="path to chromadb",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=DEFAULT_HF_EMBED_MODEL,
        help="Embedding model path",
    )

    parser.add_argument("--data-path", type=str, help="Path to json files with unstructured chunks")
    parser.add_argument("--output", help="output directory")
    args = parser.parse_args()

    settings = chromadb.get_settings()
    settings.allow_reset = True
    print(f"creating/loading db at {args.path_to_db}...")
    db = chromadb.PersistentClient(path=args.path_to_db, settings=settings)
    print("Done!")

    if args.embedding_model_path.startswith("http"):
        print(f"Using Embedding API model endpoint: {args.embedding_model_path}")
        embed_model = OpenAIEmbedding(api_base=args.embedding_model_path, api_key="dummy", embedding_ctx_length=512, chunk_size=32, tiktoken_enabled=False, embed_batch_size=32)
    else:
        print(f"Loading {args.embedding_model_path}...")
        embed_model = HuggingFaceEmbedding(args.embedding_model_path)
    
    main(args.data_path, args.path_to_db, embed_model, db)
