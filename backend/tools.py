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

import json_repair
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
            # Fire standard HTTPS web call directly to the engine
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
        return json_repair.loads(result)

    if text_blocks[0].get("snippet"):
        result=text_blocks[0].get("snippet")
        print("result(snippet):",result)
        return json_repair.loads(result)
    if text_blocks[0].get("paragraph"):
        result=text_blocks[0].get("paragraph")
        print("result(para):",result)
        return json_repair.loads(result)

    
    # print("======================================\n text block is: ",text_blocks[0].get("code"))
    # print(text_blocks)
    # print("=============================\ntext:::::::::::",text_blocks)
    # return json_repair.loads(result)
    # return [{'type': 'code_block', 'language': 'json', 'code': '{\n"research_summary": {\n"company": "EmbarkingOnVoyage",\n"market": "India",\n"audiences": ["CTO", "VP", "Digital Head"],\n"research_date": "2026-09-10",\n"methodology_note": "Focused on enterprise B2B search intent for AI-native engineering, legacy modernization, and digital transformation targeting technical decision-makers in India and international markets."\n},\n"top_opportunities": [\n{\n"rank": 1,\n"keyword": "how to implement agentic ai in enterprise systems",\n"normalized_topic": "Agentic AI Enterprise Implementation",\n"search_intent": "commercial investigation",\n"buyer_stage": "solution evaluation",\n"target_persona": "CTO",\n"business_problem": "Integrating autonomous multi-agent systems into legacy architecture without security or scaling failures.",\n"why_a_decision_maker_would_search_it": "Evaluating architectural readiness and execution risk for autonomous workflows.",\n"eov_relevance": "Direct alignment with EOV\'s core competencies in Agentic AI and LLM orchestration.",\n"business_value_score": 95,\n"buyer_intent_score": 90,\n"trend_score": 98,\n"ranking_opportunity_score": 85,\n"overall_priority_score": 92,\n"recommended_content_type": "architecture guide",\n"suggested_blog_angle": "A practical blueprint for CTOs to deploy autonomous AI agents safely in production enterprise environments.",\n"related_keyword_cluster": ["enterprise agentic ai architecture", "multi-agent orchestration framework"],\n"india_relevance": "High demand among India-based global delivery tech leaders modernizing client stacks.",\n"international_relevance": "High global demand for scalable agentic frameworks.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 2,\n"keyword": "legacy .net migration to cloud native microservices",\n"normalized_topic": "Legacy .NET Modernization",\n"search_intent": "commercial investigation",\n"buyer_stage": "solution evaluation",\n"target_persona": "VP of Engineering",\n"business_problem": "Monolithic .NET applications slowing down release cycles and increasing infrastructure costs.",\n"why_a_decision_maker_would_search_it": "Looking for step-by-step strategies to refactor or rewrite legacy .NET apps into microservices on Azure.",\n"eov_relevance": "Leverages EOV\'s Microsoft/Azure ecosystem and legacy modernization expertise.",\n"business_value_score": 90,\n"buyer_intent_score": 88,\n"trend_score": 85,\n"ranking_opportunity_score": 90,\n"overall_priority_score": 89,\n"recommended_content_type": "implementation guide",\n"suggested_blog_angle": "Minimizing downtime during complex .NET Core migration to Azure microservices.",\n"related_keyword_cluster": ["modernize .net monolith", "azure microservices migration path"],\n"india_relevance": "Very high in Indian IT hubs for enterprise legacy updates.",\n"international_relevance": "Universal enterprise requirement.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 3,\n"keyword": "ai-driven test automation for enterprise software ROI",\n"normalized_topic": "AI-Driven Test Automation ROI",\n"search_intent": "commercial investigation",\n"buyer_stage": "solution evaluation",\n"target_persona": "Head of Quality Engineering",\n"business_problem": "High regression testing overhead and slow release pipelines in large engineering teams.",\n"why_a_decision_maker_would_search_it": "Justifying budget for AI-based test generation and self-healing test scripts.",\n"eov_relevance": "Direct match for EOV\'s quality assurance and AI-driven test automation capabilities.",\n"business_value_score": 88,\n"buyer_intent_score": 85,\n"trend_score": 90,\n"ranking_opportunity_score": 88,\n"overall_priority_score": 87,\n"recommended_content_type": "ROI/business case",\n"suggested_blog_angle": "How AI-driven test automation cuts regression cycle times by 60% in enterprise applications.",\n"related_keyword_cluster": ["self healing test automation ai", "intelligent qa automation metrics"],\n"india_relevance": "Strong focus in offshore/global delivery centers based in India.",\n"international_relevance": "High global QA cost-reduction focus.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 4,\n"keyword": "how to choose an ai native product engineering partner",\n"normalized_topic": "AI-Native Partner Selection",\n"search_intent": "transactional/vendor selection",\n"buyer_stage": "vendor evaluation",\n"target_persona": "Head of Digital",\n"business_problem": "Sifting through traditional IT outsourcers who lack true AI-native development capabilities.",\n"why_a_decision_maker_would_search_it": "Establishing evaluation criteria for selecting a specialized AI product engineering firm.",\n"eov_relevance": "Direct match for EOV\'s core positioning as an AI-native engineering firm.",\n"business_value_score": 98,\n"buyer_intent_score": 95,\n"trend_score": 88,\n"ranking_opportunity_score": 70,\n"overall_priority_score": 87,\n"recommended_content_type": "vendor-selection guide",\n"suggested_blog_angle": "The 5 critical competencies to look for when hiring an AI-native engineering vendor.",\n"related_keyword_cluster": ["evaluating ai engineering vendors", "ai product development company criteria"],\n"india_relevance": "Helps position EOV locally for enterprise transformations.",\n"international_relevance": "Attracts global clients looking for India-based engineering partners.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 5,\n"keyword": "llm orchestration frameworks comparison for enterprise",\n"normalized_topic": "LLM Orchestration Comparison",\n"search_intent": "commercial investigation",\n"buyer_stage": "solution evaluation",\n"target_persona": "CTO",\n"business_problem": "Choosing between LangChain, LlamaIndex, and custom orchestration layers for enterprise data security.",\n"why_a_decision_maker_would_search_it": "Technical comparison to avoid vendor lock-in and latency bottlenecks in generative AI apps.",\n"eov_relevance": "Ties into EOV\'s LLM orchestration and enterprise software engineering services.",\n"business_value_score": 85,\n"buyer_intent_score": 82,\n"trend_score": 92,\n"ranking_opportunity_score": 85,\n"overall_priority_score": 86,\n"recommended_content_type": "comparison",\n"suggested_blog_angle": "Architectural trade-offs of enterprise LLM orchestration frameworks in production.",\n"related_keyword_cluster": ["enterprise llm orchestration", "langchain vs custom llm architecture"],\n"india_relevance": "Growing adoption among Indian digital product firms.",\n"international_relevance": "High global relevance.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 6,\n"keyword": "data engineering pipeline modernization for real-time ai",\n"normalized_topic": "Data Pipeline Modernization",\n"search_intent": "informational",\n"buyer_stage": "problem identification",\n"target_persona": "VP of Engineering",\n"business_problem": "Legacy batch data pipelines fail to feed real-time context to enterprise AI models.",\n"why_a_decision_maker_would_search_it": "Understanding how to upgrade data infrastructure to support low-latency AI inference.",\n"eov_relevance": "Direct fit with EOV\'s data engineering and modern app engineering capabilities.",\n"business_value_score": 82,\n"buyer_intent_score": 78,\n"trend_score": 88,\n"ranking_opportunity_score": 86,\n"overall_priority_score": 84,\n"recommended_content_type": "strategic guide",\n"suggested_blog_angle": "Transitioning from batch to streaming data architectures for real-time AI readiness.",\n"related_keyword_cluster": ["real time data engineering ai", "modern data stack for llms"],\n"india_relevance": "Important for data-heavy enterprises in India (fintech, SaaS).",\n"international_relevance": "Universal requirement.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 7,\n"keyword": "microservices vs serverless for enterprise applications 2026",\n"normalized_topic": "Microservices vs Serverless Architecture",\n"search_intent": "commercial investigation",\n"buyer_stage": "solution evaluation",\n"target_persona": "Chief Architect",\n"business_problem": "Balancing operational overhead of Kubernetes microservices against serverless scaling limitations.",\n"why_a_decision_maker_would_search_it": "Deciding the optimal cloud architecture model for new enterprise product builds.",\n"eov_relevance": "Aligns with EOV\'s cloud and modern application engineering capabilities.",\n"business_value_score": 80,\n"buyer_intent_score": 80,\n"trend_score": 82,\n"ranking_opportunity_score": 88,\n"overall_priority_score": 82,\n"recommended_content_type": "comparison",\n"suggested_blog_angle": "When to choose microservices vs. serverless in modern cloud application design.",\n"related_keyword_cluster": ["serverless cost vs microservices", "enterprise cloud architecture strategy"],\n"india_relevance": "Highly relevant for Indian engineering teams scaling cloud apps.",\n"international_relevance": "Global architectural debate.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 8,\n"keyword": "intelligent automation roadmap for financial services enterprises",\n"normalized_topic": "Intelligent Automation Roadmap",\n"search_intent": "commercial investigation",\n"buyer_stage": "solution evaluation",\n"target_persona": "Head of Digital Transformation",\n'}, {'type': 'paragraph', 'snippet': '"business_problem": "Siloed process automation tools failing to deliver end-to-end operational efficiency.",\n"why_a_decision_maker_would_search_it": "Looking for structured automation roadmaps combining AI agents and standard RPA/workflows.",\n"eov_relevance": "Ties into EOV\'s intelligent automation and enterprise digital transformation services.",\n"business_value_score": 86,\n"buyer_intent_score": 82,\n"trend_score": 80,\n"ranking_opportunity_score": 80,\n"overall_priority_score": 82,\n"recommended_content_type": "strategic guide",\n"suggested_blog_angle": "Building an enterprise intelligent automation roadmap that moves beyond basic RPA.",\n"related_keyword_cluster": ["intelligent automation strategy", "ai workflow automation enterprise"],\n"india_relevance": "High traction in Indian banking and financial services sector.",\n"international_relevance": "Strong global financial sector interest.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 9,\n"keyword": "enterprise azure migration best practices for .net systems",\n"normalized_topic": "Azure Migration Best Practices",\n"search_intent": "informational",\n"buyer_stage": "implementation",\n"target_persona": "VP of Engineering",\n"business_problem": "Managing cost overruns and security misconfigurations during Azure cloud migration.",\n"why_a_decision_maker_would_search_it": "Seeking architectural governance and execution guidelines for Azure migration.",\n"eov_relevance": "Direct match for EOV\'s Microsoft/Azure ecosystem and .NET engineering capabilities.",\n"business_value_score": 78,\n"buyer_intent_score": 75,\n"trend_score": 78,\n"ranking_opportunity_score": 88,\n"overall_priority_score": 79,\n"recommended_content_type": "architecture guide",\n"suggested_blog_angle": "A governance framework for moving mission-critical .NET systems to Azure securely.",\n"related_keyword_cluster": ["azure cloud migration governance", ".net enterprise azure architecture"],\n"india_relevance": "Massive enterprise adoption of Azure in India.",\n"international_relevance": "Global enterprise cloud standard.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n},\n{\n"rank": 10,\n"keyword": "digital product UX metrics that impact enterprise software adoption",\n"normalized_topic": "Enterprise UX Adoption Metrics",\n"search_intent": "informational",\n"buyer_stage": "problem identification",\n"target_persona": "Head of Product",\n"business_problem": "Low user adoption rates of internal or customer-facing enterprise software due to poor UX.",\n"why_a_decision_maker_would_search_it": "Understanding how modern UX engineering correlates with productivity and ROI.",\n"eov_relevance": "Connects to EOV\'s digital product experience / UX services.",\n"business_value_score": 75,\n"buyer_intent_score": 70,\n"trend_score": 75,\n"ranking_opportunity_score": 85,\n"overall_priority_score": 76,\n"recommended_content_type": "executive guide",\n"suggested_blog_angle": "Measuring the invisible cost of bad enterprise UX and how to fix it.",\n"related_keyword_cluster": ["enterprise UX strategy", "measuring enterprise software usability"],\n"india_relevance": "Increasing focus among Indian product engineering companies.",\n"international_relevance": "Global product priority.",\n"evidence_sources": [\n{\n"title": "EmbarkingOnVoyage Capabilities",\n"url": "https://embarkingonvoyage.com/",\n"reason": "Core service alignment",\n"date": "2026-01-01"\n}\n]\n}\n],\n"recommended_top_10": [\n"how to implement agentic ai in enterprise systems",\n"legacy .net migration to cloud native microservices",\n"ai-driven test automation for enterprise software ROI",\n"how to choose an ai native product engineering partner",\n"llm orchestration frameworks comparison for enterprise",\n"data engineering pipeline modernization for real-time ai",\n"microservices vs serverless for enterprise applications 2026",\n"intelligent automation roadmap for financial services enterprises",\n"enterprise azure migration best practices for .net systems",\n"digital product UX metrics that impact enterprise software adoption"\n]\n}'}, {'type': 'code_block', 'code': "\n\nIf you'd like, let me know:\n* Would you like me to expand these into **detailed content briefs** with H2/H3 outlines?\n* Should we adjust the **geographic weighting** between the Indian domestic market and international export markets?\n\nLet me know how you'd like to proceed with the content roadmap execution.\n\n"}]
# result=search_top_keywords(TOP_KEYWORD_QUERY)

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
        return json_repair.loads(result)

    if text_blocks[0].get("snippet"):
        result=text_blocks[0].get("snippet")
        print("result links (snippet):",result)
        return json_repair.loads(result)
    if text_blocks[0].get("paragraph"):
        result=text_blocks[0].get("paragraph")
        print("result links (para):",result)
        return json_repair.loads(result)    
    print("===========================================\nresult in links: ",result)
    # return json_repair.loads(result)
    # print("=============================\ntext:::::::::::",text_blocks)
    # return text_blocks



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
