from vanna.chromadb import ChromaDB_VectorStore
from vanna.base import VannaBase
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import EmbeddingFunction
from openai import OpenAI
import os
import ollama

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "gpt-4o-mini"

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def name(self) -> str:
        return "ollama-embedding-function"

    def __call__(self, input):
        response = ollama.embed(model=EMBED_MODEL, input=input)
        return response["embeddings"]

# -----------------------------
# CUSTOM VANNA CLASS
# -----------------------------
class MyVanna(ChromaDB_VectorStore, VannaBase):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        VannaBase.__init__(self, config=config)

    def generate_embedding(self, data: str, **kwargs):
        response = ollama.embed(
            model=EMBED_MODEL,
            input=data
        )
        return response["embeddings"][0]

    def submit_prompt(self, prompt, **kwargs) -> str:
        # If prompt is already a list of messages, pass directly
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages
        )
        return response.choices[0].message.content

    def system_message(self, message: str):
        return {"role": "system", "content": message}

    def user_message(self, message: str):
        return {"role": "user", "content": message}

    def assistant_message(self, message: str):
        return {"role": "assistant", "content": message}

# -----------------------------
# INITIALIZE VANNA
# -----------------------------
vn = MyVanna(config={
    "path": "./vanna_chroma",
    "embedding_function": OllamaEmbeddingFunction(),
    "n_results": 10,
})

# -----------------------------
# CONNECT POSTGRES
# -----------------------------
vn.connect_to_postgres(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)