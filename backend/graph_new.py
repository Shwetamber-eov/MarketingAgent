"""
LangGraph workflow for keyword -> professional, SEO-optimized blog post
generation, using structured JSON output at every step (via Pydantic schemas
+ Gemini's structured-output / function-calling mode).

Pipeline:
    keyword
      -> plan_node            (JSON: title, meta_description, slug, outline, tags)
      -> gather_links_node    (no LLM schema call - Google Search grounding via
                                the raw google-genai client; finds real external
                                sources + a real embarkingonvoyage.com page)
      -> draft_node           (JSON: sections[] with heading + content + visual flags)
      -> visual_prompts_node  (JSON: refined image/diagram prompts, gemini-3.5-flash-lite)
      -> polish_node          (JSON: final BlogPost - the complete structured blog)
      -> seo_check_node       (pure-Python Rank Math-style scoring, no LLM call)
          -> if score >= threshold or max attempts reached: END
          -> else: seo_fix_node (LLM revises only the failing checks) -> seo_check_node
      -> END

Every LLM call returns a validated Pydantic object (not raw text), so the
final state is guaranteed well-formed JSON that's easy to render, store,
or send to an API.

Notes on the SEO scoring:
    This pipeline has no live WordPress/Rank Math connection, so
    `calculate_seo_score` approximates the core checks Rank Math's analyzer
    runs (focus keyword in title/meta/slug/first-section/subheading, keyword
    density, title/meta length, content length, structural richness, tag
    count, internal/outbound links) using only the data this pipeline
    produces. It won't be byte-for-byte identical to a live Rank Math score,
    but it targets the same signals and is tuned to land in the 80+ range
    when checks pass.

Notes on visuals:
    `draft_node` flags which sections would benefit from a diagram/image and
    writes a rough one-line idea for each. `visual_prompts_node` then expands
    those rough ideas into full, production-ready image-generation prompts
    using a fixed model (gemini-3.5-flash-lite), stored in state["visual_prompts"].
    This node ONLY generates prompt text - it does not call any image
    generation API. Wiring an actual image-gen node that consumes
    state["visual_prompts"] (one entry per flagged section) is a natural
    next step and was intentionally left out for now.

Notes on links (external + internal):
    `gather_links_node` runs right after `plan_node` and populates
    state["external_links"] / state["internal_links"] with REAL (title, url)
    pairs found via Google Search grounding - NOT model-generated JSON.

    Why a separate node instead of asking the writer LLM to "search and cite"
    in one structured call: Google's API rejects requests that combine the
    google_search tool with controlled generation (response_schema /
    function-calling structured output) on most models - see the long
    comment above `_grounded_search`. So this pipeline keeps grounding calls
    schema-free and hands the resulting URLs to the (unrelated) structured
    writer calls as plain context, which they're told to copy verbatim.

    Internal links point at real embarkingonvoyage.com (EOV) pages: first
    via a `site:embarkingonvoyage.com <keyword>` grounded search, falling
    back to an LLM pick from EOV_SERVICE_CATALOG (a small, hand-verified
    list of EOV's real service pages) if that search comes back empty.

    Links are written into section content as plain Markdown
    `[anchor text](url)`, the same way the pipeline already handles bullets/
    tables/quotes. Use `markdown_links_to_html()` at render/publish time to
    turn them into `<a href="...">` tags.
"""

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

# Raw google-genai client - used ONLY for Google Search-grounded calls (see
# `_grounded_search`). Kept separate from the langchain_google_genai wrapper
# used everywhere else, since grounding + structured output don't mix.
from google import genai
from google.genai import types as google_types
from datetime import datetime
from pathlib import Path
import sys

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
            "Full section content in Markdown-safe plain text. Formatting rules "
            "this field MUST follow: "
            "(1) PARAGRAPHS: write in short paragraphs of 3-4 sentences, each "
            "separated by a blank line ('\\n\\n'). Never write a block of text "
            "longer than 4 sentences without breaking it into a new paragraph. "
            "(2) LISTS: whenever the content naturally involves 3+ items, steps, "
            "tips, features, or examples, format them as a Markdown bullet list "
            "('- item') or numbered list ('1. item') instead of cramming them "
            "into a sentence with commas. "
            "(3) TABLES FOR DIFFERENCES: whenever the section compares two or "
            "more items, options, tools, plans, or approaches across shared "
            "attributes (e.g. 'X vs Y', pros and cons, before/after, pricing "
            "tiers, feature comparisons), format that comparison as a Markdown "
            "table with a header row ('| Header | Header |') and a separator "
            "row ('|---|---|'). The table MUST contain at least 5 data rows "
            "(not counting the header or separator row) - if you don't "
            "naturally have 5 rows of real comparison points, add more "
            "attributes/criteria to compare rather than submitting a short "
            "table. Never bury a real comparison inside a paragraph or "
            "bullet list, and never pad a table with filler/repetitive rows "
            "just to hit the count. Skip this for sections with nothing to "
            "compare, and never fabricate a comparison that isn't there. "
            "(4) QUOTES: if a statistic, strong claim, or standout takeaway fits "
            "the section, set it off as a Markdown block quote ('> text') - do "
            "not force one into every section, and never invent a statistic or "
            "attribute a quote to a real named person. "
            "(5) LINKS: when a claim in this section is backed by one of the "
            "external reference links you were given, or when EOV's services "
            "(from the internal reference links you were given) are genuinely "
            "relevant to what the section discusses, embed it inline as a "
            "Markdown link ('[anchor text](URL)'). Use ONLY a URL from the "
            "reference links you were given - never invent, guess, or modify "
            "a URL. When you link to an EOV service, phrase it as a concrete "
            "benefit to the reader (what EOV's service actually does for "
            "them), not just a bare mention. Don't force a link into a "
            "section with nothing relevant to link to, and don't repeat the "
            "same link in multiple sections. "
            "A well-formed section mixes these elements as needed; it is not "
            "just a wall of prose."
        )
    )
    needs_visual: bool = Field(
        default=False,
        description=(
            "Whether this section would meaningfully benefit from an "
            "accompanying diagram or image - e.g. it describes a system/"
            "architecture, a step-by-step process or flow, or a comparison "
            "that a diagram would clarify better than text alone. Most "
            "sections do NOT need one; only flag sections where a visual "
            "adds real understanding (typically 0-2 per post)."
        ),
    )
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


_VALID_MODELS = {"gemini-3.5-flash-lite", "gemma-4-31b-it", "gemma-3-12b-it"}

# Fixed model used specifically for expanding rough visual ideas into full
# image-generation prompts - kept separate from GOOGLE_MODEL since this is a
# smaller, distinct task from the main writing pipeline.
# VISUAL_PROMPT_MODEL = "gemma-4-31b-it"
VISUAL_PROMPT_MODEL = "gemini-3.5-flash-lite"

# Fixed model used for Google Search-grounded link discovery. Confirmed to
# support the built-in `google_search` tool. Kept separate from GOOGLE_MODEL
# for the same reason as VISUAL_PROMPT_MODEL - distinct, smaller task.
SEARCH_MODEL = "gemini-3.5-flash-lite"

EOV_DOMAIN = "embarkingonvoyage.com"
EXTERNAL_LINKS_PER_POST = 5
MIN_TABLE_ROWS = 5


def get_llm(
    temperature: float = 0.7, model_override: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """
    Builds the Gemini/Gemma chat model.

    Reads the model name from the GOOGLE_MODEL env var (defaults to
    "gemini-3.5-flash-lite" if unset) so you can point this at whichever of
    the supported model strings you want without touching code. Pass
    model_override to pin a specific call to a different model regardless of
    the env var (used by the visual-prompt node, which always uses a smaller
    Gemma model).
    """
    model_name = model_override or os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment."
        )
    if model_name not in _VALID_MODELS:
        # Not fatal - Google may add new model strings after this was written -
        # but flag it since a typo'd model name is a common silent failure mode.
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
    # Uses function-calling / tool-mode structured output under the hood,
    # which Gemini/Gemma models support via langchain_google_genai. The return
    # value is a validated instance of `schema`, not raw text.
    return llm.with_structured_output(schema)





def _format_links_for_prompt(links: List[Dict[str, Any]]) -> str:
    if not links:
        return "(none found for this post - do not fabricate any links in this category)"
    formatted = []
    for link in links:
        # Extract raw URL from Markdown format:
        # [https://example.com](https://example.com)
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

# ---------------------------------------------------------------------------
# Shared SEO guidance, injected into plan/draft/polish/fix prompts
# ---------------------------------------------------------------------------
SEO_GUIDELINES = """\
This content must be optimized to score 80+ on a Rank Math-style SEO \
analysis. Use the exact FOCUS KEYWORD given to you, verbatim (same wording/ \
casing - do not swap in a synonym or a different grammatical form):

1. The focus keyword must appear in the SEO title, ideally within the first \
   half of the title.
2. The focus keyword must appear in the meta description.
3. The focus keyword must appear naturally within the first 10% of the \
   article's body content (i.e. early in the first section).
4. The focus keyword must appear in at least one subheading (section heading).
5. Keyword density across the full article should land between 0.6% and \
   2.0% of total words - enough to be findable, never stuffed.
6. The SEO title should be 50-60 characters long.
7. The meta description should be 120-160 characters long.
8. The URL slug must be a short, lowercase, hyphenated version of the title \
   that includes the focus keyword.
9. The article must contain at least one outbound link to a reputable \
   external source and at least one internal link to embarkingonvoyage.com.
"""

# ---------------------------------------------------------------------------
# Node config
# ---------------------------------------------------------------------------
LENGTH_WORDS = {
    "short": "200-250",
    "medium": "300-500",
    "long": "600-800",
}

MIN_WORDS_BY_LENGTH = {
    "short": 300,
    "medium": 600,
    "long": 900,
}

SEO_SCORE_THRESHOLD = 80
MAX_SEO_FIX_ATTEMPTS = 2

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

Example of a section that correctly uses a TABLE because it compares options - note \
it has 5 data rows, the required minimum \
(for a section about "cushioned vs minimalist running shoes"):

Choosing between cushioned and minimalist shoes comes down to how your feet \
currently handle impact, not personal preference alone. The table below lays \
out how the two styles differ on the factors that matter most.

| Factor | Cushioned | Minimalist |
|---|---|---|
| Heel-to-toe drop | 8-12mm | 0-4mm |
| Best for | Long-distance, road running | Short runs, strength-focused training |
| Injury risk if switching too fast | Low | Higher without a gradual transition |
| Typical price range | $120-$180 | $90-$140 |
| Break-in period | Minimal | 2-4 weeks, gradual |

Most runners are better off starting cushioned and transitioning gradually if \
they want to try minimalist shoes.

Example of naturally citing an external source and linking to a relevant EOV \
service with a concrete benefit (use ONLY the URLs you are actually given for \
the real post - these two are illustrative placeholders, not real ones to reuse):

Teams that skip structured gait analysis entirely see far higher return rates \
on running shoes, according to [industry retail research](https://example.com/research). \
If your team is building a fitting tool like this in-house, EOV's \
[AI-Native Digital Product Consulting](https://embarkingonvoyage.com/services/ainative-digital-product-consulting/) \
service specializes in mapping a user journey like this before writing a \
single line of code, which is usually the fastest way to avoid costly rework \
later.
"""

def _summarize_keyword_strategy(strategy: dict, max_items: int = 6) -> str:
    """
    Compresses the keyword strategy into a compact, prompt-friendly block.
    Caps each list so the plan prompt stays cheap - we only need enough
    signal to steer the LLM, not the full keyword dump.
    """
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
                "You are a professional content strategist and SEO "
                "specialist. Given a focus keyword and a keyword research "
                "report, produce a blog plan - title, meta description, "
                "URL slug, section outline (3-6 headings, at least 1 "
                "section suited for a visual such as an architecture or "
                "dataflow diagram), and tags - engineered to score 80+ on "
                "a Rank Math-style SEO analysis.\n\n"
                "Use the keyword research report as follows:\n"
                "- Adapt one of the candidate titles, or write a better "
                "one, working in the primary keyword.\n"
                "- Fold secondary keywords into the meta description and "
                "headings naturally (no stuffing).\n"
                "- Turn a couple of the question keywords into "
                "subheadings or an FAQ-style section where relevant.\n"
                "- Use semantic/related terms to add topical depth across "
                "the outline.\n"
                "- Let the content angles guide the overall narrative "
                "framing.\n\n"
                "{seo_guidelines}",
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
    # with PLAN_PATH.open("w") as json_file:
    #     json.dump({"plan": plan.model_dump()}, json_file, indent=4)
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
    word_target = LENGTH_WORDS.get(length, LENGTH_WORDS["short"])
    plan = state["plan"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional blog writer. Write clear, engaging, "
                "well-structured content for each section heading provided. "
                "Combined, all sections should total roughly {word_target} words.\n\n"

                "{seo_guidelines}\n\n"

                "Weave the exact focus keyword '{keyword}' naturally into the "
                "content - it MUST appear within the FIRST section (the first "
                "10% of the article) - and keep overall keyword density "
                "between 0.6% and 2.0% of total words. Never stuff it "
                "unnaturally.\n\n"

                "You MUST follow these formatting rules in every section:\n\n"

                "1. SHORT PARAGRAPHS - Write 3-4 sentences, then insert a blank "
                "line and start a new paragraph. Do not write a single block of "
                "5+ sentences under any circumstances. A section is normally "
                "2-4 short paragraphs, not one long one.\n\n"

                "2. BULLETS WHEN LISTING - If you are describing 3 or more items, "
                "steps, tips, features, or examples, you MUST format them as a "
                "Markdown bullet list ('- item') or numbered list ('1. item'). "
                "Do not describe a list of items inside a paragraph using commas. "
                "Skip this rule for sections that genuinely have nothing to list.\n\n"

                "3. TABLES FOR DIFFERENCES - whenever a section compares two or "
                "more items, options, tools, plans, or approaches across shared "
                "attributes (e.g. 'X vs Y', pros/cons, before/after, pricing "
                "tiers, feature comparisons), format that comparison as a "
                "Markdown table with a header row and a separator row. The "
                "table MUST have at least 5 data rows (not counting the header/ "
                "separator) - find at least 5 real attributes or criteria to "
                "compare rather than submitting a short table, but never pad "
                "with filler or repetitive rows just to hit the count. Never "
                "bury a real comparison in a paragraph or bullet list. Skip "
                "this for sections with nothing to compare.\n\n"

                "4. ONE BLOCK QUOTE WHEN IT FITS - If a section contains a "
                "statistic, a strong claim, or a summarizing takeaway, set it "
                "off using a Markdown block quote ('> text'). Do not force a "
                "quote into a section where nothing warrants it, and never "
                "invent a statistic or attribute a statement to a real named "
                "person.\n\n"

                "5. LINKS - You are given a list of real EXTERNAL reference "
                "links and a list of real INTERNAL (embarkingonvoyage.com / "
                "EOV) reference links below. Where a claim in a section is "
                "genuinely backed by one of the external links, or where one "
                "of EOV's services is genuinely relevant to what a section "
                "discusses, embed it inline as a Markdown link "
                "('[anchor text](URL)'), using the URL EXACTLY as given - "
                "never invent, guess, or alter a URL, and never link to "
                "anything not in these lists. When you link to an EOV "
                "service, phrase it around a concrete benefit to the reader "
                "(what that service actually does for them), not a bare "
                "mention. Aim to naturally use 2-3 of the external links and "
                "at least 1 of the internal links across the whole post - "
                "spread across different sections, never more than one link "
                "per section, and never force a link into a section with "
                "nothing relevant to link to.\n\n"

                "External reference links (only these URLs, or none):\n"
                "{external_links_block}\n\n"
                "Internal EOV reference links (only these URLs, or none):\n"
                "{internal_links_block}\n\n"

                "For EACH section, also decide whether it needs an "
                "accompanying visual: set needs_visual=true ONLY if the "
                "section describes a system/architecture, a step-by-step "
                "process or flow, or a comparison that a diagram would "
                "meaningfully clarify (typically 0-2 sections per post, not "
                "every section). When true, set visual_type and write a "
                "brief visual_idea; otherwise leave needs_visual=false, "
                "visual_type='none', visual_idea=''.\n\n"

                "Do not add a heading of your own - the section heading is "
                "already provided separately, and it must match the outline "
                "exactly.\n\n"
                "not necessary to add bullet points, tables, links, or a "
                "visual in every section.\n"
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
    """
    For every section the draft flagged as needing a visual, generate a
    refined, production-ready image/diagram-generation prompt.

    Uses a fixed model (gemini-3.5-flash-lite) regardless of GOOGLE_MODEL, since
    prompt-writing for image generation is a distinct, smaller task from the
    main writing pipeline. This node ONLY produces prompt text - it does not
    call any image-generation API. A future node can consume
    state["visual_prompts"] to actually generate images.
    """
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
                "You write detailed, production-ready prompts for an image/"
                "diagram-generation model, based on a rough idea. For each "
                "section provided, expand its rough visual_idea into a full "
                "image_prompt: describe every element, label, and connector "
                "that should appear, and specify a clean, professional "
                "visual style suited to a blog post titled '{title}'. Keep "
                "the visual_type as given for each section. Return one "
                "entry per section, in the same order.",
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

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional editor and SEO specialist. You are "
                "given a blog plan and drafted sections. Improve clarity, "
                "flow, grammar, and professionalism of every section's "
                "content while preserving meaning and structure.\n\n"

                "{seo_guidelines}\n\n"

                "The focus keyword is '{keyword}' - keep it present in the "
                "title, meta description, slug, at least one subheading, and "
                "the first section, at a natural density of 0.6%-2.0%. Never "
                "remove existing keyword instances unless there is clear "
                "stuffing.\n\n"

                "CRITICAL - do not flatten formatting: the drafts may already "
                "contain short paragraphs, Markdown bullet lists ('- item'), "
                "Markdown tables (each with at least 5 data rows), Markdown "
                "links ('[text](url)'), or block quotes ('> text'). You must "
                "PRESERVE these - never merge a bullet list, table, or block "
                "quote back into a plain paragraph, never drop a data row "
                "from a table below the 5-row minimum, and never strip or "
                "rewrite a Markdown link's URL. Keep the 3-4 sentence "
                "paragraph breaks intact. Never change a section's heading "
                "text - it must match the original exactly.\n\n"

                "If a section is a wall of prose with no formatting and it "
                "contains 3+ listable items, convert that list into a "
                "Markdown bullet list as part of your edit. If a section "
                "describes a comparison or difference between two or more "
                "things without a table, convert it into a Markdown table "
                "with at least 5 data rows instead. If a section contains a "
                "statistic or standout takeaway with no block quote, you may "
                "add one - but never invent a statistic or attribute a quote "
                "to a real named person. NEVER add a new link of your own - "
                "only the links already present in the draft (or, if truly "
                "needed, one of the reference links below) may appear.\n\n"

                "Reference links available if a section still needs one "
                "(use the URL EXACTLY as given, or not at all):\n"
                "External: {external_links_block}\n"
                "Internal (EOV): {internal_links_block}\n\n"

                "Refine the URL slug if needed (lowercase, hyphenated, "
                "contains the focus keyword). Write a short, strong "
                "conclusion (4-5 sentences, plain prose, no bullets, tables, "
                "or quotes). Estimate reading time in minutes from total "
                "word count (assume ~150 words/minute). Return the complete "
                "finished blog post as structured data.",
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

    # Preserve the draft's visual-need assessment exactly - that decision was
    # already finalized before visual_prompts_node ran off of it, so polish
    # should not silently redecide (or drop) needs_visual/visual_type/idea.
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
    """
    Returns a list of tables found in `text`, each represented as its list of
    data-row lines (header and separator rows excluded). Used to check the
    "at least 5 data rows per table" rule without an LLM call.
    """
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
    """
    Approximates the core checks a Rank Math-style SEO analysis runs, using
    only the data this pipeline produces (there's no live WordPress/Rank
    Math connection here). Returns a report dict with a 0-100 score, a
    per-check breakdown, and a list of failed checks with plain-language fix
    instructions suitable for feeding straight back into an LLM revision
    prompt (see seo_fix_node).
    """

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
    """
    Feeds the specific failed checks from seo_check_node back into an LLM,
    asking it to make the smallest edits necessary to fix them - rather than
    a full rewrite - then hands control back to seo_check_node to re-score.
    """
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

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an SEO editor. You are given a finished blog post "
                "(as JSON) and a specific list of SEO issues to fix. Make "
                "the smallest edits necessary to fix EVERY listed issue "
                "while preserving the post's meaning, tone, and existing "
                "Markdown formatting (paragraphs, bullet lists, tables with "
                "at least 5 data rows, links, block quotes). Never change "
                "section headings. Never invent statistics or attribute "
                "quotes to real named people. If you need to add a link to "
                "satisfy a fix, use ONLY a URL from the reference links "
                "below - never invent or modify a URL. Return the complete "
                "corrected blog post as structured data.\n\n"
                "Reference links available if a fix requires adding one:\n"
                "External: {external_links_block}\n"
                "Internal (EOV): {internal_links_block}",
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
    """
    Converts every Markdown link '[anchor text](url)' in `text` into an HTML
    `<a href="...">` tag. Call this at render/publish time - e.g. right
    before pushing a section's `content` (or the `conclusion`) into a CMS,
    HTML template, or WordPress/Rank Math field - rather than anywhere
    earlier in the pipeline, since every node up to this point deliberately
    keeps writing plain Markdown links (exactly the way it already keeps
    bullets/tables/quotes as Markdown).

    Example:
        >>> markdown_links_to_html("See [EOV's services](https://embarkingonvoyage.com/services/data-engineering/) for more.")
        'See <a href="https://embarkingonvoyage.com/services/data-engineering/" target="_blank" rel="noopener noreferrer">EOV\\'s services</a> for more.'
    """
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
    # graph.add_node("gather_links", gather_links_node)
    graph.add_node("draft", draft_node)
    graph.add_node("visual_prompts", generate_visual_prompts_node)
    graph.add_node("polish", polish_node)
    graph.add_node("seo_check", seo_check_node)
    graph.add_node("seo_fix", seo_fix_node)

    graph.set_entry_point("plan")
    # graph.add_edge("plan", "gather_links")
    # graph.add_edge("gather_links", "draft")
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
    """Convenience wrapper: run the full graph for a keyword and return state.

    state["final_blog"] is a JSON-serializable dict shaped like BlogPost:
        {
          "title": str,
          "meta_description": str,
          "slug": str,
          "tags": [str, ...],
          "sections": [
              {
                "heading": str,
                "content": str,   # Markdown, incl. "[anchor](url)" links -
                                   # run through markdown_links_to_html() at
                                   # render time to get <a href="..."> tags
                "needs_visual": bool,
                "visual_type": str,
                "visual_idea": str,
              },
              ...
          ],
          "conclusion": str,
          "estimated_read_time_minutes": int
        }

    state["external_links"] / state["internal_links"] are the real,
    search-grounded (title, url) pairs made available to the writer/editor
    nodes:
        [{"title": str, "url": str}, ...]

    state["visual_prompts"] is a list of refined image-generation prompts,
    one per section flagged needs_visual=True:
        [{"heading": str, "visual_type": str, "image_prompt": str}, ...]

    state["seo_report"] is the final Rank Math-style score breakdown:
        {"score": int, "checks": {...}, "failed_checks": [...],
         "internal_links_found": [...], "external_links_found": [...], ...}
    """
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