import argparse
from pprint import pprint
from textwrap import dedent
from typing import Optional

import chromadb
import torch
import pandas as pd
import os
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openllm import OpenLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores.types import (MetadataFilter,
                                                  MetadataFilters)
from transformers import AutoTokenizer, PreTrainedTokenizer

from rag._defaults import DEFAULT_HF_CHAT_MODEL, DEFAULT_HF_EMBED_MODEL, DEFAULT_SYTEM_PROMPT


def get_llama3_1_instruct_str(
    query: str,
    nodes: list[NodeWithScore],
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = DEFAULT_SYTEM_PROMPT,
) -> str:
    context_str = ""
    for node in nodes:
        #print(f"Context: {node.metadata}")
        context_str += node.text.replace("\n", "  \n")
    #print(f"\nUsing {context_str=}\n")

    # https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3
    context_and_query = f"""
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dedent(context_and_query).strip("\n")},
    ]
    toks = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print(f"Prefix: {len(toks)=}")
    return tokenizer.decode(toks)


def get_local_llm(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
    use_4bit_quant: bool,
    generate_kwargs: dict,
) -> HuggingFaceLLM:
    print(f"Using HF model: {model_name}")

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if use_4bit_quant:
        if not torch.cuda.is_available():
            raise ValueError("--use-4bit-quant requires a GPU")
        from transformers import BitsAndBytesConfig

        model_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        }
    else:
        model_kwargs = {"torch_dtype": torch.bfloat16}
    llm = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        generate_kwargs=generate_kwargs,
        max_new_tokens=max_new_tokens,
        stopping_ids=[tokenizer.eos_token_id],
        model_kwargs=model_kwargs,
    )
    pprint(f"Loaded model {model_name}")
    return llm


def load_data(
    embedding_model_path: str, path_to_db: str
) -> tuple[VectorStoreIndex, chromadb.GetResult]:
    if embedding_model_path.startswith("http"):
        print(f"\nUsing Embedding API model endpoint: {embedding_model_path}\n")
        embed_model = OpenAIEmbedding(api_base=embedding_model_path, api_key="dummy")
    else:
        print(f"\nEmbedding model: {embedding_model_path}\n")
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_path)
    chroma_client = chromadb.PersistentClient(path_to_db)
    chroma_collection = chroma_client.get_collection(name="documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index, chroma_collection.get()


def create_retriever(cutoff: float, top_k_retriever: int, filters=None) -> VectorIndexRetriever:
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k_retriever,
        filters=filters,
        node_postprocessors=[SimilarityPostprocessor(similarity=cutoff)],
    )
    return retriever


def get_nodes(
    query: str, retriever: VectorIndexRetriever, reranker: Optional[LLMRerank] = None
) -> list[NodeWithScore]:
    """
    Retrieve the most relevant chunks, given the query.
    """
    # Wrap in a QueryBundle class in order to use reranker.
    query_bundle = QueryBundle(query)
    nodes = retriever.retrieve(query_bundle)

    if reranker is not None:
        nodes = reranker.postprocess_nodes(nodes, query_bundle)
    return nodes

def get_llm_answer(llm, tag, args, query_list=None):

    filters=None
    if tag:
        filters = MetadataFilters(filters=[MetadataFilter(key="Tag", value=tag)], condition="or")
    elif args.folder:
        filters = MetadataFilters(filters=[MetadataFilter(key="Tag", value=args.folder)], condition="or")
    
    retriever = create_retriever(
        cutoff=args.cutoff, top_k_retriever=args.top_k_retriever, filters=filters
    )
    
    d = {}
    d["Queries"] = []
    d["Answers"] = []
    d["Main Source"] = []
    
    if query_list:
        d["Queries"] = query_list
        
        for q in query_list:
            print("\nQuery: " + q)
            nodes = get_nodes(q, retriever, reranker)
            prefix = get_llama3_1_instruct_str(q, nodes, tokenizer)
            
            output_response = llm.complete(prefix)
            print(f"{output_response.text=}\n")
            
            d["Answers"].append(output_response.text)
            d["Main Source"].append(nodes[0].node.metadata["Source"] + ", page " + str(nodes[0].node.metadata["Page Number"]))
    else:
        d["Queries"] = [args.query]
        
        nodes = get_nodes(args.query, retriever, reranker)
        prefix = get_llama3_1_instruct_str(args.query, nodes, tokenizer)
        #print(f"\n{prefix=}\n")
        output_response = llm.complete(prefix)
        print(f"\n{output_response.text=}\n")
        
        d["Answers"].append(output_response.text)
        d["Main Source"].append(nodes[0].node.metadata["Source"] + ", page " + str(nodes[0].node.metadata["Page Number"]))

    if args.output_folder:
        output_df = pd.DataFrame(data=d)
        
        suffix = "all_documents_mixed"
        if tag:
            suffix = tag
        elif args.folder:
            suffix = args.folder
            
        csv_name = args.output_folder + "/extracted_info_" + suffix +".csv"
        print("Saving output to " + csv_name)
        output_df.to_csv(csv_name, index=False)
        

def print_references(nodes):
    # TODO: @garrett.goon - Delete below, just for debugging/visuals
    print("\n **** REFERENCES **** \n")
    for n in nodes[0:1]:
        title = n.node.metadata["Source"]
        page = n.node.metadata["Page Number"]
        text = n.node.text
        commit = n.node.metadata["Commit"]
        doctag = n.node.metadata["Tag"]
        newtext = text.encode("unicode_escape").decode("unicode_escape")
        out_title = f"**Source:** {title}  \n **Page:** {page}  \n **Similarity Score:** {round((n.score * 100),3)}% \n"
        out_text = f"**Text:**  \n {newtext}  \n"

        print(f"\n{out_title=}")
        print(f"{out_text=}\n")


if __name__ == "__main__":
    print("\n**********  QUERYING **********\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="What is the meaning of life?", help="Query to ask of the llm")
    parser.add_argument("--path-to-db", type=str, default="db", help="path to chroma db")
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=DEFAULT_HF_EMBED_MODEL,
        help="local path or URL to sentence transformer model",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_HF_CHAT_MODEL,
        help="local path or URL to chat model",
    )
    parser.add_argument(
        "--chat-model-endpoint",
        default=None,
        type=str,
        help="HTTP path to model endpoint, if serving",
    )
    parser.add_argument(
        "--top-k-retriever",
        default=5,
        type=int,
        help="top k results for retriever",
    )
    parser.add_argument(
        "--top-k-reranker",
        default=None,
        type=int,
        help="top k results for reranker",
    )
    parser.add_argument(
        "--temp",
        default=0.2,
        type=float,
        help="Generation temp",
    )
    parser.add_argument(
        "--top-p",
        default=0.9,
        type=float,
        help="top p probability for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=250,
        type=int,
        help="Max generation toks",
    )
    parser.add_argument(
        "--cutoff",
        default=0.6,
        type=float,
        help="Filter out docs with score below cutoff.",
    )
    parser.add_argument(
        "--use-4bit-quant",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--streaming",
        help="stream responses",
        action="store_true",
    )
    parser.add_argument(
        "--folder",
        default=None,
        type=str,
        help="Only use documents initially under that folder name.",
    )
    parser.add_argument(
        "--query-all",
        help="Query documents from each folder separately. Overrides the --folder argument.",
        action="store_true",
    )
    parser.add_argument(
        "--query-file",
        default=None,
        type=str,
        help="txt file containing a single query per line. Overrides the --query argument.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        type=str,
        help="Save queries output to csv files under that path.",
    )
    
    
    args = parser.parse_args()
    if "Meta-Llama-3.1" not in args.model_name:
        # Only tested with Meta-Llama-3.1 so far. The system prompt and tokenization would need to
        # be adjusted for other models.
        raise ValueError(f"Script expects a Llama-3.1 model, not {args.model_name}")

    if args.top_k_reranker and args.top_k_reranker > args.top_k_retriever:
        raise ValueError("top_k_reranker, if provided, must be smaller than top_k_retriever.")

    index, chunks = load_data(args.embedding_model_path, args.path_to_db)
    
    
    tags = []
    for i in range(len(chunks["ids"])):
        eltags = chunks["metadatas"][i]["Tag"]
        if eltags not in tags:
            tags.append(eltags)
    print("\nTags: " + str(tags) + "\n")
    
    
    reranker = LLMRerank(top_n=args.top_k_reranker) if args.top_k_reranker else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    generate_kwargs = {
        "do_sample": True,
        "temperature": args.temp,
        "top_p": args.top_p,
    }
    if args.chat_model_endpoint:
        print(f"\nUsing hosted LLM at: {args.chat_model_endpoint}\n")
        llm = OpenLLM(
            model=args.model_name,
            api_base=args.chat_model_endpoint,
            api_key="fake",
            generate_kwargs=generate_kwargs,
            max_tokens=args.max_new_tokens,
        )
    else:
        print(f"\nUsing local {args.model_name} LLM\n")
        llm = get_local_llm(
            args.model_name, tokenizer, args.max_new_tokens, args.use_4bit_quant, generate_kwargs
        )
    
    if args.output_folder and not os.path.exists(args.output_folder):
        print(args.output_folder + " does not exist yet, creating it...")
        os.makedirs(args.output_folder)
    
    
    # Get the list of queries from the queries file
    query_list = None
    if args.query_file:
        print("Using " + args.query_file + " as query list")
        query_file = open(args.query_file, "r")
        query_lines = query_file.readlines()
        query_file.close()
        
        query_list = []
        for query in query_lines:
            query_list.append(query.replace("\n",""))
    
    # Loop though all folders if wanting to get query answers for all docs
    if args.query_all:
        for tag in tags:
            print("\n\nApply query to " + tag + " folder only")
            get_llm_answer(llm, tag, args, query_list)
    
    # Otherwise, either get answers either from all documents mixed up, or from a single folder if using --folder parameter
    else:
        get_llm_answer(llm, None, args, query_list)

