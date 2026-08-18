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
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
load_dotenv()

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
        description=(
            "Full section content in Markdown-safe plain text. Formatting rules "
            "this field MUST follow: "
            "(1) PARAGRAPHS: write in short paragraphs of 3-4 sentences, each "
            "separated by a blank line ('\\n\\n'). Never write a block of text "
            "longer than 4 sentences without breaking it into a new paragraph. "
            "(2) LISTS: whenever the content naturally involves 3+ items, steps, "
            "tips, features, or examples, format them as a Markdown bullet list "
            "('- item') or numbered list ('1. item') instead of cramming them "
            "into a sentence with commas. "
            "(3) QUOTES: if a statistic, strong claim, or standout takeaway fits "
            "the section, set it off as a Markdown block quote ('> text') - do "
            "not force one into every section, and never invent a statistic or "
            "attribute a quote to a real named person. "
            "A well-formed section mixes these elements; it is not just a wall "
            "of prose."
        )
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
    conclusion: str = Field(
        description=(
            "A short, strong closing paragraph (3-4 sentences max). Plain "
            "prose, no bullet points or block quotes here."
        )
    )
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
# Model strings currently valid on the Gemini API (generativelanguage.googleapis.com)
# for GOOGLE_MODEL. Gemma models are served through the same Gemini API/key, so no
# separate credential is needed to switch between these.
#   "gemini-3.5-flash-lite"  -> Google's hosted lite model. Best instruction-following
#                                of the three for a nuanced formatting task like this,
#                                and the cheapest/fastest. Recommended default.
#   "gemma-4-31b-it"         -> Open-weight dense 31B model, native function calling.
#                                Noticeably stronger reasoning/instruction-following
#                                than gemma-3-12b-it, at higher latency/cost than
#                                Flash-Lite.
#   "gemma-3-12b-it"         -> Open-weight 12B model. Fastest/cheapest of the open
#                                models but the weakest at following multi-part
#                                formatting instructions (bullets/quotes/paragraph
#                                length) under structured/function-calling output -
#                                expect more misses on this task than the other two.
_VALID_MODELS = {"gemini-3.5-flash-lite", "gemma-4-31b-it", "gemma-3-12b-it"}


def get_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """
    Builds the Gemini/Gemma chat model.

    Reads the model name from the GOOGLE_MODEL env var (defaults to
    "gemini-3.5-flash-lite" if unset) so you can point this at whichever of
    the supported model strings you want without touching code.
    """
    model_name = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment."
        )
    if model_name not in _VALID_MODELS:
        # Not fatal - Google may add new model strings after this was written -
        # but flag it since a typo'd model name is a common silent failure mode.
        print(
            f"[warning] GOOGLE_MODEL={model_name!r} is not one of the models this "
            f"workflow was tuned against ({sorted(_VALID_MODELS)}). Proceeding anyway."
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
    # which Gemini/Gemma models support via langchain_google_genai. The return
    # value is a validated instance of `schema`, not raw text.
    return llm.with_structured_output(schema)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
LENGTH_WORDS = {
    "short": "200-250",
    "medium": "300-500",
    "long": "600-800",
}

# A concrete worked example of the expected formatting. Small/lite models follow
# a demonstrated pattern far more reliably than an abstract list of rules, so this
# is included directly in the draft prompt.
FORMATTING_EXAMPLE = """\
Example of correctly formatted section content (for a section about "choosing a \
running shoe"):

Picking the right running shoe comes down to matching the shoe to your gait and \
mileage, not just the brand. Most runners fall into one of three categories: \
neutral, overpronator, or supinator. Getting a gait analysis at a specialty running \
store is the fastest way to find out which one you are.

Once you know your gait type, a few features matter more than the rest:

- **Cushioning**: more cushioning reduces impact on long runs but can feel less responsive
- **Drop**: the heel-to-toe height difference, usually 0-12mm
- **Stability**: added support for overpronators, usually a firmer foam wedge

> Runners who replace shoes every 300-500 miles report noticeably fewer overuse \
injuries than those who run shoes into the ground.

Try on shoes later in the day, when your feet are slightly swollen, and always \
walk or jog a few steps in-store before buying.
"""


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
                "Combined, all sections should total roughly {word_target} words.\n\n"

                "You MUST follow these three formatting rules in every section:\n\n"

                "1. SHORT PARAGRAPHS - Write 3-4 sentences, then insert a blank "
                "line and start a new paragraph. Do not write a single block of "
                "5+ sentences under any circumstances. A section is normally "
                "2-4 short paragraphs, not one long one.\n\n"

                "2. BULLETS WHEN LISTING - If you are describing 3 or more items, "
                "steps, tips, features, or examples, you MUST format them as a "
                "Markdown bullet list ('- item') or numbered list ('1. item'). "
                "Do not describe a list of items inside a paragraph using commas. "
                "Skip this rule for sections that genuinely have nothing to list.\n\n"

                "3. ONE BLOCK QUOTE WHEN IT FITS - If a section contains a "
                "statistic, a strong claim, or a summarizing takeaway, set it "
                "off using a Markdown block quote ('> text'). Do not force a "
                "quote into a section where nothing warrants it, and never "
                "invent a statistic or attribute a statement to a real named "
                "person.\n\n"

                "Do not add a heading of your own - the section heading is "
                "already provided separately.\n\n"
                "not necessary to add bullet points in every section"
                "{formatting_example}",
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
            "formatting_example": FORMATTING_EXAMPLE,
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
                "meaning and structure.\n\n"

                "CRITICAL - do not flatten formatting: the drafts may already "
                "contain short paragraphs, Markdown bullet lists ('- item'), or "
                "block quotes ('> text'). You must PRESERVE these - never merge "
                "a bullet list or block quote back into a plain paragraph, and "
                "never merge separate short paragraphs into one long paragraph. "
                "Keep the 3-4 sentence paragraph breaks intact.\n\n"

                "If a section is a wall of prose with no formatting and it "
                "contains 3+ listable items, convert that list into a Markdown "
                "bullet list as part of your edit. If a section contains a "
                "statistic or standout takeaway with no block quote, you may "
                "add one - but never invent a statistic or attribute a quote "
                "to a real named person.\n\n"

                "Write a short, strong conclusion (4-5 sentences, plain prose, "
                "no bullets or quotes). Estimate reading time in minutes from "
                "total word count (assume ~150 words/minute). Return the "
                "complete finished blog post as structured data.",
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
    length: str = "short",
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