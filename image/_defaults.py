from textwrap import dedent

DEFAULT_HF_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_HF_CHAT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CHUNK_STRAT = "by_title"
DEFAULT_HF_CHAT_TEMPLATE = "\n<|user|>:{}</s>\n<|assistant|>:"
DEFAULT_SYTEM_PROMPT = dedent("""
You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
""").strip("\n")
# The embedding model hosted on houston errors out at larger batch sizes
DEFAULT_MAX_EMBED_BSZ = 32
