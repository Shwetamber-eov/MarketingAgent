import os
from typing import Optional
from pathlib import Path
import sys
import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8001))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = "seo_rag1"

GEMMA_MODEL = os.getenv("GEMMA_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TOP_K_DEFAULT = 5  # how many nearest existing posts Chroma hands to the LLM

# ---------------------------------------------------------------------------
# Vector store — unchanged embedding backend (Ollama nomic-embed-text), since
# the existing collection is already embedded with this model. Swapping the
# embedding function requires re-ingesting everything (dimensions must match).
# ---------------------------------------------------------------------------
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
chroma_http_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
vector_store = Chroma(
    client=chroma_http_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# ---------------------------------------------------------------------------
# Ingestion — load / refresh blog_posts.csv into the Chroma collection.
# Uses the post URL as a stable id, so re-running this upserts rather than
# creating duplicate vectors when the CSV is refreshed.
# ---------------------------------------------------------------------------
def ingest_blogs_to_chroma(csv_path: str = "blog_posts.csv", batch_size: int = 100) -> int:
    df = pd.read_csv(csv_path)

    documents, ids, metadatas = [], [], []
    for _, row in df.iterrows():
        title = str(row.get("title", "") or "").strip()
        if not title:
            continue
        excerpt = str(row.get("excerpt", "") or "").strip()
        url = str(row.get("url", "") or "").strip()

        documents.append(f"{title}. {excerpt}".strip())
        ids.append(url or title)  # stable unique id enables upsert on re-run
        metadatas.append(
            {
                "title": title,
                "url": url,
                "date": str(row.get("date", "") or ""),
                "author": str(row.get("author", "") or ""),
            }
        )

    for i in range(0, len(documents), batch_size):
        vector_store.add_texts(
            texts=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
        print("document: ",i)

    print(f"Ingested {len(documents)} blog posts into Chroma collection '{COLLECTION_NAME}'.")
    return len(documents)

ROOT = Path(__file__).resolve().parent.parent.parent

csv_path=ROOT / "blog_posts.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("file not found")
rows=df["title"].tolist()
ingest_blogs_to_chroma()