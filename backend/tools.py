import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import markdown
import requests
import os
from typing import Optional
from dotenv import load_dotenv

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from backend.variables.prompts import TOP_KEYWORD_QUERY, GATHER_LINKS_QUERY
load_dotenv()

import serpapi
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8002))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11435")
COLLECTION_NAME = "seo_rag1"

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b-it")
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
# Gemini — the final-level judge
# ---------------------------------------------------------------------------
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "Set GOOGLE_API_KEY (or GEMINI_API_KEY) to your Gemini API key before running this."
    )

judge_llm = ChatGoogleGenerativeAI(
    model=GEMMA_MODEL,
    temperature=0,  # deterministic classification, not creative generation
    google_api_key=GOOGLE_API_KEY,
)

class DuplicateVerdict(BaseModel):
    is_duplicate: bool = Field(
        description="True only if an existing post already fully covers the same "
        "core intent and meaning as the new keyword/topic — not just a related subject."
    )
    matched_title: str = Field(
        default="", description="Title of the closest matching existing post, if is_duplicate is True."
    )
    matched_url: str = Field(
        default="", description="URL of the closest matching existing post, if is_duplicate is True."
    )
    confidence: float = Field(description="Confidence in this verdict, from 0.0 to 1.0.")
    reasoning: str = Field(description="One or two sentences explaining the decision.")


JUDGE_SYSTEM_PROMPT = """You are an editorial assistant for a content team. \
Decide whether a NEW blog topic/keyword would duplicate the core intent and meaning \
of any EXISTING blog post below, even when the exact wording or keywords differ.

Focus only on: what the reader is trying to learn or accomplish, and what problem \
or topic the post fundamentally addresses. Ignore surface wording differences.

Mark is_duplicate = true only if a reader searching for the NEW topic would already \
have their need fully met by one of the EXISTING posts.

Mark is_duplicate = false if the existing posts cover a different angle, a different \
audience, a narrower or broader scope, or a related-but-distinct subtopic — even if \
they were retrieved as the closest semantic matches.

Only consider the candidates listed below; they were pre-filtered by vector similarity, \
so a low similarity score does not necessarily mean "not a duplicate" and a high one \
does not necessarily mean "duplicate" — judge based on the actual content."""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "New blog keyword/topic:\n{new_keyword}\n\n"
            "Existing candidate posts (nearest semantic matches from the vector store):\n"
            "{candidates_block}\n\n"
            "Return your verdict.",
        ),
    ]
)

structured_judge = judge_llm.with_structured_output(DuplicateVerdict)
judge_chain = JUDGE_PROMPT | structured_judge


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def isblogexist(new_keyword: str, top_k: int = TOP_K_DEFAULT) -> tuple[bool, str, Optional[dict]]:
    """
    Checks whether a blog covering the same intent/meaning as `new_keyword`
    already exists, using Chroma for semantic retrieval and Gemini as the
    final arbiter of intent.

    Returns:
        (is_duplicate, reason, details)
        details = {"verdict": DuplicateVerdict, "candidates": [...]} or None
        if the vector store was empty.
    """
    results = vector_store.similarity_search_with_score(new_keyword, k=top_k)

    if not results:
        return False, "No existing blogs in the vector store to compare against.", None

    candidates = [
        {
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("url", ""),
            "excerpt": doc.page_content,
            "vector_distance": round(float(score), 4),  # lower = closer match
        }
        for doc, score in results
    ]

    candidates_block = "\n\n".join(
        f"{i + 1}. Title: {c['title']}\n"
        f"   URL: {c['url']}\n"
        f"   Excerpt: {c['excerpt'][:400]}\n"
        f"   Vector distance: {c['vector_distance']}"
        for i, c in enumerate(candidates)
    )

    verdict: DuplicateVerdict = judge_chain.invoke(
        {"new_keyword": new_keyword, "candidates_block": candidates_block}
    )

    reason = (
        f"Duplicate found (confidence {verdict.confidence:.2f}): "
        f"'{verdict.matched_title}' ({verdict.matched_url}). {verdict.reasoning}"
        if verdict.is_duplicate
        else f"No duplicate found (confidence {verdict.confidence:.2f}). {verdict.reasoning}"
    )

    return verdict.is_duplicate, reason, {"verdict": verdict, "candidates": candidates}



def send_blog_email(
    blog_markdown: str,
    sender_app_password: str = os.getenv("GOOGLE_APP_PASSWORD"),
    sender_email: str = "ronittayade1@gmail.com",
    recipient_email: str = "ronittayade1@gmail.com",
    title: str = "",
    subject: str = "Your AI Generated Blog "
):
    """
    Send the generated blog Markdown as an email.
    Args:
        recipient_email: User's email address.
        blog_markdown: Markdown string generated by blog_json_to_markdown().
        sender_email: Gmail address used to send the email.
        sender_app_password: Gmail App Password.
        subject: Email subject.
    """
    print("inside send mail func")
    blog_html = markdown.markdown(
        blog_markdown,
        extensions=[
            "extra",
            "nl2br",
            "sane_lists"
        ]
    )
    msg = MIMEMultipart("alternative")

     # Convert Markdown → HTML
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject + title

    # Plain-text version
    # msg.attach(MIMEText(blog_markdown, "plain", "utf-8"))

    #html version
    msg.attach(MIMEText(blog_html, "html", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_app_password)
            server.sendmail(
                sender_email,
                recipient_email,
                msg.as_string()
            )
        return True

    except Exception as e:
        print(f":::::::::::::::::::::::::::::::::\nFailed to send email: {e}\n:::::::::::::::::::::::::::::::::::::::::::")
        return False


def search_top_keywords(query):
    SERP_API_KEY=os.getenv("SERP_API_KEY")
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_ai_mode",
        "q": query,
        "api_key": SERP_API_KEY,
        # India-focused search
        "location": "India",
        "gl": "in",

        # English-language decision-maker searches
        "hl": "en",

        # Structured response for your LangGraph/parser
        "output": "json",

        # Fresh research rather than cached result
        "no_cache": True,
    }
    
    try:
            # Fire standard HTTPS web call directly to the engine
        response = requests.get(url, params=params)
            
        if response.status_code != 200:
            print(f"❌ SerpApi server rejected query. Code: {response.status_code}")
            print(f"Server message: {response.text}")
            return []
        results = response.json()
    except Exception as e:
        print(f"Workflow execution pipeline failed: {e}")
        return []
    # print("result:::::::", results)
    text_blocks = results["text_blocks"]
    print(type(text_blocks))
    # print(text_blocks)
    # print("=============================\ntext:::::::::::",text_blocks)
    return text_blocks

# result=search_top_keywords(TOP_KEYWORD_QUERY)



# if __name__ == "__main__":
#     # Run once (or whenever blog_posts.csv is refreshed) to (re)populate the vector store:
#     # ingest_blogs_to_chroma("blog_posts.csv")

#     keyword = "benefits of forward deployment engineering for enterprise software adoption"
#     exists, reason, details = isblogexist(keyword)

#     print(f"\nKeyword: {keyword}")
#     print(f"Exists:  {exists}")
#     print(f"Reason:  {reason}")
#     if details:
#         print("\nCandidates considered:")
#         for c in details["candidates"]:
#             print(f"  - {c['title']} (distance={c['vector_distance']}) -> {c['url']}")
