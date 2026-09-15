import json
import os
import re
import re
from typing import Dict, Any
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import time

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from datetime import datetime
from pathlib import Path
import sys

from backend.variables.prompts import (
    SYSTEM_PROMPT_DRAFT, 
    SYSTEM_PROMPT_PLAN, 
    SYSTEM_PROMPT_IMAGE, 
    SYSTEM_PROMPT_POLISH, 
    SYSTEM_PROMPT_FIX,
    SEO_GUIDELINES,
    VISUAL_PROMPT_MODEL,
    _VALID_MODELS,
    )
from backend.variables.structured_output_description import SECTION_CONTENT_DESCRIPTION
from backend.variables.guidelines import (
    SEO_GUIDELINES, 
    FORMATTING_EXAMPLE,
    MIN_TABLE_ROWS,
    EOV_DOMAIN,
    MAX_SEO_FIX_ATTEMPTS,
    MIN_WORDS_BY_LENGTH,
    SEO_SCORE_THRESHOLD,
    LENGTH_WORDS)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Get current date and time
now = datetime.now()

# Format as Hour:Minute:Second
current_time = now.strftime("%H:%M:%S")
print("Current Time:", current_time)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PLAN_PATH = BASE_DIR / "backend" / "test" / "plan.json"
GATHER_LINKS_PATH = BASE_DIR / "backend" / "test" / "gather_links.json"
DRAFT_PATH = BASE_DIR / "backend" / "test" / "draft.json"
POLISH_PATH = BASE_DIR / "backend" / "test" / "polish.json"
SEO_FIX_PATH = BASE_DIR / "backend" / "test" / "seo_fix.json"
SEO_CHECK_PATH = BASE_DIR / "backend" / "test" / "seo_check.json"

PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
class BlogPlan(BaseModel):
    title: str = Field(
        description=(
            "Compelling, SEO-friendly blog title, 50-60 characters long, "
            "containing the exact focus keyword (ideally within the first "
            "half of the title)."
        )
    )
    meta_description: str = Field(
        description=(
            "SEO meta description, 120-160 characters long, containing the "
            "exact focus keyword near the start, written to earn clicks from "
            "a search results page."
        )
    )
    slug: str = Field(
        description=(
            "URL-friendly slug for the post: lowercase words separated by "
            "hyphens, containing the exact focus keyword, no stop-word "
            "clutter, ideally under 75 characters "
            "(e.g. 'best-running-shoes-flat-feet')."
        )
    )
    outline: List[str] = Field(
        description=(
            "3 to 6 section headings, in order, covering the topic. Include "
            "the focus keyword naturally in at least one heading."
        )
    )
    tags: List[str] = Field(description="3 to 6 relevant SEO tags/keywords")


class Section(BaseModel):
    heading: str = Field(description="Section heading, matches an outline item")
    content: str = Field(
        description=(
            SECTION_CONTENT_DESCRIPTION
        ))
    needs_visual: bool = Field(
        default=False,
        description=(
            "Whether this section would meaningfully benefit from an "
            "accompanying diagram or image - e.g. it describes a system/"
            "architecture, a step-by-step process or flow, or a comparison "
            "that a diagram would clarify better than text alone. Most "
            "sections do NOT need one; only flag sections where a visual "
            "adds real understanding (typically 0-2 per post)."
        ),)
    visual_type: str = Field(
        default="none",
        description=(
            "Required if needs_visual is True, otherwise 'none'. One of: "
            "'architecture_diagram' (system/component structure), "
            "'flow_diagram' (sequential process/steps/decision flow), "
            "'comparison_chart' (visual side-by-side of options), "
            "'illustration' (conceptual/explanatory image), 'none'."
        ),
    )
    visual_idea: str = Field(
        default="",
        description=(
            "Required if needs_visual is True, otherwise empty string. A "
            "brief 1-2 sentence rough description of what the visual should "
            "depict for this section. This is a rough idea only - it will be "
            "expanded into a full image-generation prompt in a later step, "
            "so keep it concise and focused on content, not style."
        ),
    )


class BlogSections(BaseModel):
    sections: List[Section] = Field(
        description="One entry per outline heading, in the same order"
    )


class RefinedVisualPrompt(BaseModel):
    heading: str = Field(
        description="Must exactly match the section heading this visual belongs to"
    )
    visual_type: str = Field(
        description=(
            "Carried over from the section: architecture_diagram, "
            "flow_diagram, comparison_chart, or illustration."
        )
    )
    image_prompt: str = Field(
        description=(
            "A detailed, self-contained, production-ready prompt for an "
            "image/diagram-generation model to create this visual. Describe "
            "composition, every labeled element/step/component, a clean "
            "professional visual style (e.g. 'flat-style technical diagram, "
            "white background, labeled boxes and arrows'), and any text "
            "labels that must appear. Describe the desired image directly - "
            "do not reference 'this blog post' or say 'generate an image'."
        )
    )


class VisualPromptBatch(BaseModel):
    prompts: List[RefinedVisualPrompt] = Field(
        description=(
            "One entry per section that was flagged needs_visual=True, in "
            "the same order they were given."
        )
    )


class LinkCandidate(BaseModel):
    title: str = Field(description="Exact title copied from the catalog entry given to you")
    url: str = Field(description="Exact url copied from the catalog entry given to you - never invented")


class InternalLinkSelection(BaseModel):
    picks: List[LinkCandidate] = Field(
        description=(
            "The 1-2 catalog entries most relevant to the blog topic, with "
            "title/url copied verbatim from the catalog - never invented or "
            "modified."
        )
    )


class BlogPost(BaseModel):
    title: str
    meta_description: str
    slug: str = Field(
        description=(
            "Final URL-friendly slug: lowercase, hyphenated, containing the "
            "exact focus keyword."
        )
    )
    tags: List[str]
    sections: List[Section]
    conclusion: str = Field(
        description=(
            "A short, strong closing paragraph (3-4 sentences max). Plain "
            "prose, no bullet points, tables, or block quotes here."
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
    keyword_strategy: dict
    tone: str
    audience: str
    length: str          # "short" | "medium" | "long"
    plan: dict              # BlogPlan.model_dump()
    external_links: list      # [{"title": str, "url": str}, ...] - real, search-grounded
    internal_links: list      # [{"title": str, "url": str}, ...] - real embarkingonvoyage.com pages
    sections: list           # list[dict] from BlogSections
    visual_prompts: list      # list[dict] from VisualPromptBatch - [{"heading","visual_type","image_prompt"}, ...]
    final_blog: dict         # BlogPost.model_dump() -- the final structured output
    seo_report: dict         # calculate_seo_score() output for the current final_blog
    seo_fix_attempts: int    # how many times seo_fix_node has run
    error: str



def get_llm(
    temperature: float = 0.7, model_override: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    model_name = model_override or os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment."
        )
    if model_name not in _VALID_MODELS:
        print(
            f"[warning] model={model_name!r} is not one of the models this "
            f"workflow was tuned against ({sorted(_VALID_MODELS)}). Proceeding anyway."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def get_structured_llm(
    schema: type[BaseModel],
    temperature: float = 0.7,
    model_override: Optional[str] = None,
):
    """Returns an LLM bound to always respond with the given Pydantic schema."""
    llm = get_llm(temperature=temperature, model_override=model_override)
    return llm.with_structured_output(schema)


def _format_links_for_prompt(links: List[Dict[str, Any]]) -> str:
    if not links:
        return "(none found for this post - do not fabricate any links in this category)"
    formatted = []
    for link in links:
        raw_url = link.get("url", "")
        markdown_match = re.match(r"\[.*?\]\((.*?)\)", raw_url)

        if markdown_match:
            raw_url = markdown_match.group(1)

        title = link.get("title", "")
        anchor = link.get("suggested_anchor_text", "")
        why_relevant = (
            link.get("why_relevant")
            or link.get("why_authoritative")
            or ""
        )
        # Internal/company links
        if link.get("page_type") or link.get("linking_purpose"):
            formatted.append(
                f"- Title: {title}\n"
                f"  URL: {raw_url}\n"
                f"  Suggested anchor: {anchor}\n"
                f"  Why relevant: {why_relevant}"
            )
        # External/reference links
        else:
            publisher = link.get("publisher", "")
            claim = link.get("claim_supported", "")
            formatted.append(
                f"- Title: {title}\n"
                f"  URL: {raw_url}\n"
                f"  Publisher: {publisher}\n"
                f"  Suggested anchor: {anchor}\n"
                f"  Claim supported: {claim}\n"
                f"  Why authoritative: {why_relevant}"
            )
    return "\n".join(formatted)


def _summarize_keyword_strategy(strategy: dict, max_items: int = 6) -> str:
    if not strategy:
        return "No keyword strategy provided."

    def _fmt(label, items):
        items = (items or [])[:max_items]
        return f"- {label}: {', '.join(items)}" if items else f"- {label}: (none)"

    return "\n".join([
        f"- Search intent: {strategy.get('search_intent', 'n/a')}",
        _fmt("Primary keywords", strategy.get("primary_keywords")),
        _fmt("Secondary keywords", strategy.get("secondary_keywords")),
        _fmt("Long-tail keywords", strategy.get("long_tail_keywords")),
        _fmt("Question keywords (use as subheadings/FAQ)", strategy.get("question_keywords")),
        _fmt("Semantic/related terms (weave in naturally)", strategy.get("semantic_keywords")),
        _fmt("Candidate titles", strategy.get("blog_title_ideas")),
        _fmt("Content angles", strategy.get("content_angles")),
    ])


def plan_node(state: BlogState) -> BlogState:
    print("==============================\n\n in planning node \n\n=====================================")
    # now = datetime.now()
    # print("Current Time:", now.strftime("%H:%M:%S"))
    # start_time = time.perf_counter()

    keyword_strategy = state.get("keyword_strategy") or {}
    primary_keywords = keyword_strategy.get("primary_keywords") or []
    # Prefer an explicit keyword if given, else fall back to the top primary keyword
    focus_keyword = state.get("keyword") or (primary_keywords[0] if primary_keywords else "")
    keyword_context = _summarize_keyword_strategy(keyword_strategy)

    structured_llm = get_structured_llm(BlogPlan, temperature=0.6)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT_PLAN,
            ),
            (
                "human",
                "Focus keyword: {keyword}\n"
                "Target audience: {audience}\n"
                "Tone: {tone}\n\n"
                "Keyword research report:\n{keyword_context}",
            ),
        ]
    )
    chain = prompt | structured_llm
    plan: BlogPlan = chain.invoke(
        {
            "keyword": focus_keyword,
            "audience": state.get("audience", "general readers"),
            "tone": state.get("tone", "professional"),
            "seo_guidelines": SEO_GUIDELINES,
            "keyword_context": keyword_context,
        }
    )
    with PLAN_PATH.open("w") as json_file:
        json.dump({"plan": plan.model_dump()}, json_file, indent=4)
    # print(time.perf_counter() - start_time)
    return {"plan": plan.model_dump()}


def draft_node(state: BlogState) -> BlogState:
    print("==============================\n\n in draft node \n\n=====================================")
    now = datetime.now()

# Format as Hour:Minute:Second
    current_time = now.strftime("%H:%M:%S")
    print("Current Time:", current_time)
    start_time = time.perf_counter()
    structured_llm = get_structured_llm(BlogSections, temperature=0.7)
    length = state.get("length", "medium")
    word_target = LENGTH_WORDS.get(length, LENGTH_WORDS["medium"])
    plan = state["plan"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT_DRAFT,
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
            "seo_guidelines": SEO_GUIDELINES,
            "external_links_block": _format_links_for_prompt(state.get("external_links", [])),
            "internal_links_block": _format_links_for_prompt(state.get("internal_links", [])),
        }
    )
    with DRAFT_PATH.open("w") as json_file:
            json.dump([s.model_dump() for s in result.sections], json_file, indent=4)
    print(time.perf_counter()-start_time)
    return {"sections": [s.model_dump() for s in result.sections]}


def generate_visual_prompts_node(state: BlogState) -> BlogState:
    print("==============================\n\n in generate prompt node \n\n=====================================")
    now = datetime.now()

# Format as Hour:Minute:Second
    current_time = now.strftime("%H:%M:%S")
    print("Current Time:", current_time)
    start_time = time.perf_counter()
    sections = state["sections"]
    flagged = [s for s in sections if s.get("needs_visual")]
    if not flagged:
        return {"visual_prompts": []}

    structured_llm = get_structured_llm(
        VisualPromptBatch, temperature=0.6, model_override=VISUAL_PROMPT_MODEL
    )
    plan = state["plan"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT_IMAGE,
            ),
            (
                "human",
                "Sections needing visuals:\n{sections_needing_visuals}",
            ),
        ]
    )
    sections_text = "\n\n".join(
        f"Heading: {s['heading']}\n"
        f"Visual type: {s['visual_type']}\n"
        f"Rough idea: {s['visual_idea']}"
        for s in flagged
    )
    chain = prompt | structured_llm
    result: VisualPromptBatch = chain.invoke(
        {"title": plan["title"], "sections_needing_visuals": sections_text}
    )
    print(time.perf_counter()-start_time)
    return {"visual_prompts": [p.model_dump() for p in result.prompts]}


def polish_node(state: BlogState) -> BlogState:
    print("==============================\n\n in polish node \n\n=====================================")
    now = datetime.now()

# Format as Hour:Minute:Second
    current_time = now.strftime("%H:%M:%S")
    print("Current Time:", current_time)
    start_time = time.perf_counter()
    structured_llm = get_structured_llm(BlogPost, temperature=0.4)
    plan = state["plan"]
    sections = state["sections"]
    length = state.get("length", "medium")
    word_target = LENGTH_WORDS.get(length, LENGTH_WORDS["medium"])
    print("============================================\nword target", word_target)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT_POLISH,
            ),
            (
                "human",
                "Title: {title}\n"
                "Meta description: {meta_description}\n"
                "Slug: {slug}\n"
                "Tags: {tags}\n\n"
                "Draft sections (JSON):\n{sections}",
            ),
        ]
    )
    chain = prompt | structured_llm
    post: BlogPost = chain.invoke(
        {
            "title": plan["title"],
            "word_target": word_target,
            "meta_description": plan["meta_description"],
            "slug": plan["slug"],
            "tags": ", ".join(plan["tags"]),
            "sections": json.dumps(sections, ensure_ascii=False),
            "keyword": state["keyword"],
            "seo_guidelines": SEO_GUIDELINES,
            "external_links_block": _format_links_for_prompt(state.get("external_links", [])),
            "internal_links_block": _format_links_for_prompt(state.get("internal_links", [])),
        }
    )
    post_dict = post.model_dump()

    orig_by_heading = {s["heading"]: s for s in sections}
    for sec in post_dict["sections"]:
        orig = orig_by_heading.get(sec["heading"])
        if orig:
            sec["needs_visual"] = orig["needs_visual"]
            sec["visual_type"] = orig["visual_type"]
            sec["visual_idea"] = orig["visual_idea"]
    with POLISH_PATH.open("w") as json_file:
            json.dump(post_dict, json_file, indent=4)
    print(time.perf_counter()-start_time)
    return {"final_blog": post_dict}

# ---------------------------------------------------------------------------
# Rank Math-style SEO scoring (pure Python, no LLM calls)
# ---------------------------------------------------------------------------
def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

def _keyword_occurrences(text: str, keyword: str) -> int:
    """Case-insensitive count of the exact keyword phrase in text."""
    if not keyword or not text:
        return 0
    pattern = r"\b" + re.escape(keyword.strip()) + r"\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)")

def _extract_markdown_links(text: str) -> List[tuple]:
    """Returns a list of (anchor_text, url) tuples found in Markdown-linked text."""
    return _MD_LINK_RE.findall(text or "")

_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(\s*:?-{2,}:?\s*\|)+\s*:?-{2,}:?\s*\|?\s*$")


def _extract_markdown_tables(text: str) -> List[List[str]]:
    lines = (text or "").splitlines()
    tables: List[List[str]] = []
    i = 0
    while i < len(lines) - 1:
        header, sep = lines[i].strip(), lines[i + 1].strip()
        if header.startswith("|") and _TABLE_SEPARATOR_RE.match(sep):
            data_rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                data_rows.append(lines[j])
                j += 1
            tables.append(data_rows)
            i = j
        else:
            i += 1
    return tables


def calculate_seo_score(blog: dict, keyword: str, length: str = "medium") -> dict:
    title = blog.get("title", "")
    meta = blog.get("meta_description", "")
    slug = blog.get("slug", "")
    sections = blog.get("sections", [])
    headings = [s.get("heading", "") for s in sections]
    full_body = "\n\n".join(s.get("content", "") for s in sections)
    first_section_text = sections[0].get("content", "") if sections else ""

    total_words = _word_count(full_body)
    keyword_total = _keyword_occurrences(full_body, keyword)
    density = (keyword_total / total_words * 100) if total_words else 0.0
    min_words = MIN_WORDS_BY_LENGTH.get(length, MIN_WORDS_BY_LENGTH["medium"])

    all_links = _extract_markdown_links(full_body)
    internal_links_found = [
        {"anchor": a, "url": u} for a, u in all_links if EOV_DOMAIN in u
    ]
    external_links_found = [
        {"anchor": a, "url": u} for a, u in all_links if EOV_DOMAIN not in u
    ]

    tables = _extract_markdown_tables(full_body)
    undersized_tables = [t for t in tables if len(t) < MIN_TABLE_ROWS]

    checks = {
        "title_has_keyword": {
            "weight": 10,
            "passed": _keyword_occurrences(title, keyword) > 0,
            "fix": f"Include the exact focus keyword '{keyword}' in the title.",
        },
        "title_length_ok": {
            "weight": 8,
            "passed": 50 <= len(title) <= 60,
            "fix": f"Rewrite the title to be 50-60 characters (currently {len(title)}).",
        },
        "meta_has_keyword": {
            "weight": 10,
            "passed": _keyword_occurrences(meta, keyword) > 0,
            "fix": f"Include the exact focus keyword '{keyword}' in the meta description.",
        },
        "meta_length_ok": {
            "weight": 8,
            "passed": 120 <= len(meta) <= 160,
            "fix": f"Rewrite the meta description to be 120-160 characters (currently {len(meta)}).",
        },
        "slug_has_keyword": {
            "weight": 6,
            "passed": _keyword_occurrences(slug.replace("-", " "), keyword) > 0,
            "fix": f"Update the URL slug to contain the focus keyword '{keyword}'.",
        },
        "keyword_in_first_section": {
            "weight": 12,
            "passed": _keyword_occurrences(first_section_text, keyword) > 0,
            "fix": f"Mention the exact focus keyword '{keyword}' within the first section.",
        },
        "keyword_in_subheading": {
            "weight": 10,
            "passed": any(_keyword_occurrences(h, keyword) > 0 for h in headings),
            "fix": f"Include the focus keyword '{keyword}' in at least one section heading.",
        },
        "keyword_density_ok": {
            "weight": 16,
            "passed": 0.5 <= density <= 2.5,
            "fix": (
                f"Adjust keyword usage - current density is {density:.2f}%, "
                "target 0.6%-2.0%. Add a few more natural mentions if too "
                "low, or remove some if it reads as stuffed."
            ),
        },
        "content_length_ok": {
            "weight": 12,
            "passed": total_words >= min_words,
            "fix": f"Expand the content - currently {total_words} words, target at least {min_words}.",
        },
        "has_structured_elements": {
            "weight": 8,
            "passed": any(
                ("- " in s.get("content", ""))
                or ("|" in s.get("content", ""))
                or (">" in s.get("content", ""))
                for s in sections
            ),
            "fix": "Add at least one Markdown list, table, or block quote where the content supports it.",
        },
        "tags_present": {
            "weight": 10,
            "passed": 3 <= len(blog.get("tags", [])) <= 8,
            "fix": "Provide 3-8 relevant SEO tags.",
        },
        "internal_link_present": {
            "weight": 8,
            "passed": len(internal_links_found) > 0,
            "fix": (
                f"Add at least one internal Markdown link to a relevant "
                f"{EOV_DOMAIN} page, phrased around a concrete reader "
                "benefit."
            ),
        },
        "external_link_present": {
            "weight": 6,
            "passed": len(external_links_found) > 0,
            "fix": "Add at least one outbound Markdown link to a reputable external source that supports a claim in the post.",
        },
        "tables_meet_min_rows": {
            "weight": 6,
            "passed": len(undersized_tables) == 0,
            "fix": (
                f"Every Markdown table must have at least {MIN_TABLE_ROWS} "
                "data rows (not counting the header/separator) - expand the "
                "undersized table(s) with more real comparison rows."
            ),
        },
    }

    max_score = sum(c["weight"] for c in checks.values())
    score = sum(c["weight"] for c in checks.values() if c["passed"])
    score_pct = round((score / max_score) * 100) if max_score else 0

    failed_checks = [
        {"check": name, "fix": c["fix"]} for name, c in checks.items() if not c["passed"]
    ]

    return {
        "score": score_pct,
        "checks": checks,
        "failed_checks": failed_checks,
        "keyword_density_pct": round(density, 2),
        "total_words": total_words,
        "internal_links_found": internal_links_found,
        "external_links_found": external_links_found,
    }


def seo_check_node(state: BlogState) -> BlogState:
    """Pure-Python scoring pass - no LLM call. Just measures final_blog."""
    print("==============================\n\n in seo check node \n\n=====================================")
    now = datetime.now()

# Format as Hour:Minute:Second
    current_time = now.strftime("%H:%M:%S")
    print("Current Time:", current_time)
    start_time = time.perf_counter()
    report = calculate_seo_score(
        state["final_blog"], state["keyword"], state.get("length", "medium")
    )
    with SEO_CHECK_PATH.open("w") as json_file:
            json.dump(report, json_file, indent=4)
    print(time.perf_counter()-start_time)
    return {"seo_report": report}


def seo_fix_node(state: BlogState) -> BlogState:
    print("==============================\n\n in seo fix node \n\n=====================================")
    now = datetime.now()

# Format as Hour:Minute:Second
    current_time = now.strftime("%H:%M:%S")
    print("Current Time:", current_time)
    start_time = time.perf_counter()
    structured_llm = get_structured_llm(BlogPost, temperature=0.3)
    blog = state["final_blog"]
    report = state["seo_report"]
    fixes = "\n".join(f"- {f['check']}: {f['fix']}" for f in report["failed_checks"])
    length = state.get("length", "medium")
    word_target = LENGTH_WORDS.get(length, LENGTH_WORDS["medium"])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT_FIX,
            ),
            (
                "human",
                "Focus keyword: {keyword}\n\n"
                "Issues to fix:\n{fixes}\n\n"
                "Current blog post (JSON):\n{blog_json}",
            ),
        ]
    )
    chain = prompt | structured_llm
    revised: BlogPost = chain.invoke(
        {
            "keyword": state["keyword"],
            "word_target": word_target,
            "fixes": fixes,
            "blog_json": json.dumps(blog, ensure_ascii=False),
            "external_links_block": _format_links_for_prompt(state.get("external_links", [])),
            "internal_links_block": _format_links_for_prompt(state.get("internal_links", [])),
        }
    )
    revised_dict = revised.model_dump()

    # Preserve visual-need assessment exactly, same reasoning as polish_node.
    orig_by_heading = {s["heading"]: s for s in blog.get("sections", [])}
    for sec in revised_dict["sections"]:
        orig = orig_by_heading.get(sec["heading"])
        if orig:
            sec["needs_visual"] = orig.get("needs_visual", False)
            sec["visual_type"] = orig.get("visual_type", "none")
            sec["visual_idea"] = orig.get("visual_idea", "")
    with SEO_FIX_PATH.open("w") as json_file:
        json.dump(revised_dict, json_file, indent=4)
    print(time.perf_counter()-start_time)
    return {
        "final_blog": revised_dict,
        "seo_fix_attempts": state.get("seo_fix_attempts", 0) + 1,
    }


def route_after_seo_check(state: BlogState) -> str:
    report = state.get("seo_report", {})
    attempts = state.get("seo_fix_attempts", 0)
    if report.get("score", 0) >= SEO_SCORE_THRESHOLD or attempts >= MAX_SEO_FIX_ATTEMPTS:
        return "end"
    return "fix"


# ---------------------------------------------------------------------------
# Rendering helper: Markdown links -> HTML anchors
# ---------------------------------------------------------------------------
def markdown_links_to_html(text: str, new_tab: bool = True) -> str:
    target_attr = ' target="_blank" rel="noopener noreferrer"' if new_tab else ""

    def _replace(match: "re.Match") -> str:
        anchor, url = match.group(1), match.group(2)
        return f'<a href="{url}"{target_attr}>{anchor}</a>'

    return _MD_LINK_RE.sub(_replace, text or "")


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(BlogState)
    graph.add_node("plan", plan_node)
    graph.add_node("draft", draft_node)
    graph.add_node("visual_prompts", generate_visual_prompts_node)
    graph.add_node("polish", polish_node)
    graph.add_node("seo_check", seo_check_node)
    graph.add_node("seo_fix", seo_fix_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "draft")
    graph.add_edge("draft", "visual_prompts")
    graph.add_edge("visual_prompts", "polish")
    graph.add_edge("polish", "seo_check")
    graph.add_conditional_edges(
        "seo_check", route_after_seo_check, {"end": END, "fix": "seo_fix"}
    )
    graph.add_edge("seo_fix", "seo_check")
    return graph.compile()


def generate_blog(
    keyword: str,
    keyword_strategy: dict,
    internal_links: list,
    external_links:list,
    tone: str = "professional",
    audience: str = "general readers",
    length: str = "medium",
) -> BlogState:
    app = build_graph()
    initial_state: BlogState = {
        "keyword": keyword,
        "keyword_strategy": keyword_strategy,
        "internal_links": internal_links,
        "external_links":external_links,
        "tone": tone,
        "audience": audience,
        "length": length,
        "seo_fix_attempts": 0,
    }
    return app.invoke(initial_state)