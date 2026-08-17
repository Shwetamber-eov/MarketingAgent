"""
LangGraph workflow for keyword -> professional blog post generation,
using structured JSON output at every step (via Pydantic schemas +
Gemini's structured-output / function-calling mode).

Pipeline:
    keyword
      -> plan_node    (JSON: title, meta_description, outline, tags)
      -> draft_node   (JSON: sections[] with heading + content)
      -> polish_node  (JSON: final BlogPost - the complete structured blog)
      -> END

Every LLM call returns a validated Pydantic object (not raw text), so the
final state is guaranteed well-formed JSON that's easy to render, store,
or send to an API.
"""

import os
from typing import TypedDict, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class BlogPlan(BaseModel):
    title: str = Field(description="Compelling, SEO-friendly blog title")
    meta_description: str = Field(
        description="150-160 character SEO meta description"
    )
    outline: List[str] = Field(
        description="3 to 6 section headings, in order, covering the topic"
    )
    tags: List[str] = Field(description="3 to 6 relevant SEO tags/keywords")


class Section(BaseModel):
    heading: str = Field(description="Section heading, matches an outline item")
    content: str = Field(
        description="Full section content in Markdown-safe plain text "
        "(paragraphs, may include bullet points using '-')"
    )


class BlogSections(BaseModel):
    sections: List[Section] = Field(
        description="One entry per outline heading, in the same order"
    )


class BlogPost(BaseModel):
    title: str
    meta_description: str
    tags: List[str]
    sections: List[Section]
    conclusion: str = Field(description="A short, strong closing paragraph")
    estimated_read_time_minutes: int = Field(
        description="Estimated reading time in minutes, based on total word count"
    )


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class BlogState(TypedDict, total=False):
    keyword: str
    tone: str
    audience: str
    length: str          # "short" | "medium" | "long"
    plan: dict             # BlogPlan.model_dump()
    sections: list          # list[dict] from BlogSections
    final_blog: dict        # BlogPost.model_dump() -- the final structured output
    error: str


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """
    Builds the Gemini chat model.

    Reads the model name from the GOOGLE_MODEL env var so you can point
    this at whichever Gemini Flash-Lite model string is currently valid
    for your API key (e.g. "gemini-2.0-flash-lite", "gemini-2.5-flash-lite",
    etc.) without touching code.
    """
    model_name = os.environ.get("GOOGLE_MODEL", "gemini-3.1-flash-lite")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def get_structured_llm(schema: type[BaseModel], temperature: float = 0.7):
    """Returns an LLM bound to always respond with the given Pydantic schema."""
    llm = get_llm(temperature=temperature)
    # Uses function-calling / tool-mode structured output under the hood,
    # which Gemini models support via langchain_google_genai. The return
    # value is a validated instance of `schema`, not raw text.
    return llm.with_structured_output(schema)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
LENGTH_WORDS = {
    "short": "200-250",
    "medium": "800-1200",
    "long": "1500-2000",
}


def plan_node(state: BlogState) -> BlogState:
    structured_llm = get_structured_llm(BlogPlan, temperature=0.6)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional content strategist. Given a keyword, "
                "produce a blog plan: a compelling title, an SEO meta "
                "description, a logical section outline (3-6 headings), and "
                "relevant tags.",
            ),
            (
                "human",
                "Keyword/topic: {keyword}\n"
                "Target audience: {audience}\n"
                "Tone: {tone}",
            ),
        ]
    )
    chain = prompt | structured_llm
    plan: BlogPlan = chain.invoke(
        {
            "keyword": state["keyword"],
            "audience": state.get("audience", "general readers"),
            "tone": state.get("tone", "professional"),
        }
    )
    return {"plan": plan.model_dump()}


def draft_node(state: BlogState) -> BlogState:
    structured_llm = get_structured_llm(BlogSections, temperature=0.7)
    length = state.get("length", "medium")
    word_target = LENGTH_WORDS.get(length, LENGTH_WORDS["short"])
    plan = state["plan"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional blog writer. Write clear, engaging, "
                "well-structured content for each section heading provided. "
                "Combined, all sections should total roughly {word_target} words. "
                "Write plain prose (Markdown-safe), no headings inside content "
                "since the heading is already given separately.",
            ),
            (
                "human",
                "Title: {title}\n"
                "Tone: {tone}\n"
                "Audience: {audience}\n"
                "Keyword to naturally weave in for SEO: {keyword}\n\n"
                "Section headings to write content for, in order:\n{outline}",
            ),
        ]
    )
    chain = prompt | structured_llm
    result: BlogSections = chain.invoke(
        {
            "word_target": word_target,
            "title": plan["title"],
            "outline": "\n".join(f"- {s}" for s in plan["outline"]),
            "tone": state.get("tone", "professional"),
            "audience": state.get("audience", "general readers"),
            "keyword": state["keyword"],
        }
    )
    return {"sections": [s.model_dump() for s in result.sections]}


def polish_node(state: BlogState) -> BlogState:
    structured_llm = get_structured_llm(BlogPost, temperature=0.4)
    plan = state["plan"]
    sections = state["sections"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional editor. You are given a blog plan and "
                "drafted sections. Improve clarity, flow, grammar, and "
                "professionalism of every section's content while preserving "
                "meaning and structure. Write a short, strong conclusion. "
                "Estimate reading time in minutes from total word count "
                "(assume ~200 words/minute). Return the complete finished "
                "blog post as structured data.",
            ),
            (
                "human",
                "Title: {title}\n"
                "Meta description: {meta_description}\n"
                "Tags: {tags}\n\n"
                "Draft sections (JSON):\n{sections}",
            ),
        ]
    )
    chain = prompt | structured_llm
    post: BlogPost = chain.invoke(
        {
            "title": plan["title"],
            "meta_description": plan["meta_description"],
            "tags": ", ".join(plan["tags"]),
            "sections": sections,
        }
    )
    return {"final_blog": post.model_dump()}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(BlogState)
    graph.add_node("plan", plan_node)
    graph.add_node("draft", draft_node)
    graph.add_node("polish", polish_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "draft")
    graph.add_edge("draft", "polish")
    graph.add_edge("polish", END)

    return graph.compile()


def generate_blog(
    keyword: str,
    tone: str = "professional",
    audience: str = "general readers",
    length: str = "medium",
) -> BlogState:
    """Convenience wrapper: run the full graph for a keyword and return state.

    state["final_blog"] is a JSON-serializable dict shaped like BlogPost:
        {
          "title": str,
          "meta_description": str,
          "tags": [str, ...],
          "sections": [{"heading": str, "content": str}, ...],
          "conclusion": str,
          "estimated_read_time_minutes": int
        }
    """
    app = build_graph()
    initial_state: BlogState = {
        "keyword": keyword,
        "tone": tone,
        "audience": audience,
        "length": length,
    }
    return app.invoke(initial_state)