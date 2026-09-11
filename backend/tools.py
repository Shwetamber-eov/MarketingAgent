import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import markdown
import requests
import os
from typing import Optional
from dotenv import load_dotenv

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json_repair
from backend.variables.queries import (
    TOP_KEYWORD_QUERY, 
    GATHER_LINKS_QUERY, 
    SAMPLE_KEYWORD_OUTPUT, 
    SAMPLE_LINKS_OUTPUT)
from backend.variables.prompts import JUDGE_SYSTEM_PROMPT

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8002))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11435")
COLLECTION_NAME = "seo_rag1"

# GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b-it")
GEMMA_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

TOP_K_DEFAULT = 5  # how many nearest existing posts Chroma hands to the LLM

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
    sender_email: str = os.getenv("RECIPIENT_MAIL_ID"),
    recipient_email: str = os.getenv("RECIPIENT_MAIL_ID"),
    title: str = "",
    subject: str = "Your AI Generated Blog "
):
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


def search_top_keywords():
    print("in search keyword")
    query=TOP_KEYWORD_QUERY
    SERP_API_KEY=os.getenv("SERP_API_KEY")
    print("serp: ", SERP_API_KEY)
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
        response = requests.get(url, params=params)
        print("after getting response")
        if response.status_code != 200:
            print(f"❌ SerpApi server rejected query. Code: {response.status_code}")
            print(f"Server message: {response.text}")
            return []
        results = response.json()
        # print("============================\nsearch result:", results["text_blocks"].get("code"))
    except Exception as e:
        print(f"Workflow execution pipeline failed: {e}")
        return []
    # print("result:::::::", results)
    text_blocks = results["text_blocks"]
    print(type(text_blocks))
    print(text_blocks)
    print(text_blocks[0])
    if text_blocks[0].get("code"):
        result=text_blocks[0].get("code")
        print("result (code):",result)
        # return json_repair.loads(result)
        return SAMPLE_KEYWORD_OUTPUT

    if text_blocks[0].get("snippet"):
        result=text_blocks[0].get("snippet")
        print("result(snippet):",result)
        # return json_repair.loads(result)
        return SAMPLE_KEYWORD_OUTPUT
    
    if text_blocks[0].get("paragraph"):
        result=text_blocks[0].get("paragraph")
        print("result(para):",result)
        # return json_repair.loads(result)
        return SAMPLE_KEYWORD_OUTPUT

def gather_links(query):
    print("in links")
    SERP_API_KEY=os.getenv("SERP_API_KEY")
    url = "https://serpapi.com/search"
    final_query=GATHER_LINKS_QUERY.format(TOPIC=query)
    # print(final_query[:500])
    print("=================================================\nfinal query",final_query[:500])
    params = {
        "engine": "google_ai_mode",
        "q": final_query,
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
    if text_blocks[0].get("code"):
        result=text_blocks[0].get("code")
        print("result links (code):",result)
        # return json_repair.loads(result)
        return SAMPLE_LINKS_OUTPUT

    if text_blocks[0].get("snippet"):
        result=text_blocks[0].get("snippet")
        print("result links (snippet):",result)
        # return json_repair.loads(result)
        return SAMPLE_LINKS_OUTPUT
    
    if text_blocks[0].get("paragraph"):
        result=text_blocks[0].get("paragraph")
        print("result links (para):",result)
        # return json_repair.loads(result)    
        return SAMPLE_LINKS_OUTPUT
    
    print("===========================================\nresult in links: ",result)