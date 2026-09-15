import os
import csv
from pathlib import Path

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8002))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11435")
COLLECTION_NAME = "seo_rag1"

GEMMA_MODEL = os.getenv("GEMMA_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TOP_K_DEFAULT = 5  # how many nearest existing posts Chroma hands to the LLM

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = ROOT / "data" / "blog_posts.csv"       # new posts written by the scraper
DEST_FILE = ROOT / "data" / "old_blog_posts.csv"      # full archive

# ---------------------------------------------------------------------------
# Vector store — unchanged embedding backend (Ollama nomic-embed-text), since
# the existing collection is already embedded with this model. Swapping the
# embedding function requires re-ingesting everything (dimensions must match).
#
# NOTE: we deliberately do NOT delete/recreate the collection here. Each
# document is added with a stable id (the post URL), so re-running this
# script upserts existing ids instead of duplicating them. Wiping the
# collection on every run would throw away every previously-ingested post
# each time this script runs against a CSV that only contains new posts.
# ---------------------------------------------------------------------------
embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model="nomic-embed-text")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)


# ---------------------------------------------------------------------------
# Ingestion — load blog_posts.csv (new posts since the last run) into the
# Chroma collection. Uses the post URL as a stable id, so re-running this
# upserts rather than creating duplicate vectors.
# ---------------------------------------------------------------------------
def ingest_blogs_to_chroma(csv_path: Path, batch_size: int = 100) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f"No new posts to ingest (missing or empty file: {csv_path}).")
        return 0

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"No new posts to ingest (0 rows in {csv_path}).")
        return 0

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

    if not documents:
        print(f"No valid rows to ingest in {csv_path}.")
        return 0

    for i in range(0, len(documents), batch_size):
        vector_store.add_texts(
            texts=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
        print("document batch starting at:", i)

    print(f"Ingested {len(documents)} blog posts into Chroma collection '{COLLECTION_NAME}'.")
    return len(documents)


# ---------------------------------------------------------------------------
# Archive — move the just-ingested rows from blog_posts.csv into
# old_blog_posts.csv, then clear blog_posts.csv so the next scraper run
# starts from an empty buffer.
#
# This assumes blog_scraper.py writes ONLY newly-discovered posts to
# SOURCE_FILE each run (not the full existing+new set) — otherwise rows
# already present in DEST_FILE would be appended again and duplicate.
# ---------------------------------------------------------------------------
def append_csv_and_clear_source(source_file: Path, dest_file: Path):
    if not source_file.exists() or source_file.stat().st_size == 0:
        print(f"Source file missing or empty, nothing to archive: {source_file}")
        return

    with open(source_file, "r", newline="", encoding="utf-8") as src:
        rows = list(csv.reader(src))

    if not rows:
        print("Source file is empty. Nothing to append.")
        return

    header = rows[0]
    data_rows = rows[1:]

    if not data_rows:
        print("Source file has only a header, no new rows to archive.")
        # Still fine to leave source as-is; nothing to clear.
        return

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_exists = dest_file.exists()
    dest_empty = not dest_exists or dest_file.stat().st_size == 0

    with open(dest_file, "a", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest)

        if dest_empty:
            writer.writerow(header)
            writer.writerows(data_rows)
            print("Destination was empty. Header and data copied.")
        else:
            writer.writerows(data_rows)
            print("Destination already has content. Only data rows appended.")

    # Clear source file after successful append (header + data both removed;
    # blog_scraper.py rewrites the header fresh on its next run).
    with open(source_file, "w", newline="", encoding="utf-8") as src:
        pass

    print("Source file content cleared.")
    print(f"Appended {len(data_rows)} data row(s) to {dest_file}")


if __name__ == "__main__":
    ingest_blogs_to_chroma(csv_path=SOURCE_FILE)
    append_csv_and_clear_source(SOURCE_FILE, DEST_FILE)