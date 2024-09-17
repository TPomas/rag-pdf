import argparse
from pprint import pprint
from textwrap import dedent
from typing import Optional

import chromadb
import torch
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer, PreTrainedTokenizer
from llama_index.llms.openllm import OpenLLM

from rag._defaults import DEFAULT_HF_CHAT_MODEL, DEFAULT_HF_EMBED_MODEL, DEFAULT_SYTEM_PROMPT


def get_llama3_1_instruct_str(
    query: str,
    nodes: list[NodeWithScore],
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = DEFAULT_SYTEM_PROMPT,
) -> str:
    context_str = ""
    for node in nodes:
        print(f"Context: {node.metadata}")
        context_str += node.text.replace("\n", "  \n")
    print(f"\nUsing {context_str=}\n")

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
    
    return tokenizer.decode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

def get_complete_prompt(query: str,
    nodes: list[NodeWithScore],
    system_prompt: str = DEFAULT_SYTEM_PROMPT,
) -> str:
    
    context_str = ""
    
    for node in nodes:
        print(f"Context: {node.metadata}")
        context_str += node.text.replace("\n", "  \n")
    print(f"\nUsing {context_str=}\n")
     
    instructions = dedent("""
    You are an assistant for answering questions.
    You are given the extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say "I do not know." Don't make up an answer.
    """).strip("\n")
    
    text_qa_template_str_llama3 = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Context information is
        below.
        ---------------------
        {context_str}
        ---------------------
        Using
        the context information, answer the question: {query}
        {instructions}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    return text_qa_template_str_llama3
    
    
def get_llm(
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    temp: float,
    max_new_tokens: int,
    top_p: float,
    use_4bit_quant: bool,
) -> HuggingFaceLLM:
    pprint(f"Using HF model: {model_name}")

    generate_kwargs = {
        "do_sample": True,
        "temperature": temp,
        "top_p": top_p,
    }
    model_kwargs = {"torch_dtype": torch.bfloat16}
    
    # MLIS endpoint
    if model_name.startswith("http"):
        print(f"Using OpenLLM model endpoint: {args.path_to_chat_model}")
        modelpath = str(args.path_to_chat_model)
        llm = OpenLLM(address=modelpath, generate_kwargs=generate_kwargs)
        
    # Garrett optimized code
    else:
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

def get_llm_from_endpoint(
    model_name: str,
    temp: float,
    max_new_tokens: int,
    top_p: float,
) -> OpenLLM:
    pprint(f"Using HF model: {model_name}")

    generate_kwargs = {
        "do_sample": True,
        "temperature": temp,
        "top_p": top_p,
    }
    model_kwargs = {"torch_dtype": torch.bfloat16}
    
    # MLIS endpoint
    if model_name.startswith("http"):
        print(f"Using OpenLLM model endpoint: {model_name}")
        modelpath = str(model_name)
        llm = OpenLLM(address=model_name, generate_kwargs=generate_kwargs)
    else:
        print("Expected loading model from endpoint")
    
    pprint(f"Loaded model {model_name}")
    return llm


def load_data(
    embedding_model_path: str, path_to_db: str
) -> tuple[VectorStoreIndex, chromadb.GetResult]:
    
    if embedding_model_path.startswith("http"):
        pprint(f"Using Embedding API model endpoint: {embedding_model_path}")
        embed_model = OpenAIEmbedding(api_base=embedding_model_path, api_key="dummy")
    else:
        pprint(f"Embedding model: {embedding_model_path}")
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


if __name__ == "__main__":
    print("\n**********  QUERYING **********\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Query to ask of the llm")
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
    args = parser.parse_args()
    #if "Meta-Llama-3.1" not in args.model_name:
        # Only tested with Meta-Llama-3.1 so far. The system prompt and tokenization would need to
        # be adjusted for other models.
    #    raise ValueError(f"Script expects a Llama-3.1 model, not {args.model_name}")

    if args.top_k_reranker and args.top_k_reranker > args.top_k_retriever:
        raise ValueError("top_k_reranker, if provided, must be smaller than top_k_retriever.")

    index, _ = load_data(args.embedding_model_path, args.path_to_db)
    print("DB loaded")
    
    retriever = create_retriever(
        cutoff=args.cutoff, top_k_retriever=args.top_k_retriever, filters=None
    )
    print("Retriever created")
    
    #reranker = LLMRerank(top_n=args.top_k_reranker) if args.top_k_reranker else None
    reranker = None

    nodes = get_nodes(args.query, retriever, reranker)
    print("Nodes obtained")
    
    #if not args.model_name.startswith("http"):
    #    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #prefix = get_llama3_1_instruct_str(args.query, nodes, tokenizer)
    prefix = get_complete_prompt(args.query, nodes)

    print(f"\n{prefix=}\n")
    
    

    #llm = get_llm(
    #    args.model_name,
    #    tokenizer,
    #    args.temp,
    #    args.max_new_tokens,
    #    args.top_p,
    #    args.use_4bit_quant,
    #)
    llm = get_llm_from_endpoint(args.model_name, args.temp,args.max_new_tokens, args.top_p)
    
    output_response = llm.complete(prefix)
    print(f"\n{output_response.text=}\n")

    # TODO: @garrett.goon - Delete below, just for debugging/visuals
    print("\n **** REFERENCES **** \n")
    for n in nodes:
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
