import json
from dotenv import load_dotenv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv()

# ---------------------------------------------------------------------------
# Visual-prompt helpers
# ---------------------------------------------------------------------------
VISUAL_TYPE_ICONS = {
    "architecture_diagram": "🏗️",
    "flow_diagram": "🔀",
    "comparison_chart": "📊",
    "illustration": "🎨",
}

def visual_type_label(visual_type: str) -> str:
    return (visual_type or "image").replace("_", " ").title()

def build_visual_prompt_map(visual_prompts: list) -> dict:
    """heading -> refined visual-prompt dict, for quick lookup while rendering."""
    return {vp["heading"]: vp for vp in (visual_prompts or [])}

def image_placeholder_markdown(section: dict, visual_map: dict) -> str:
    """Same placeholder, as a Markdown blockquote, for the .md export."""
    vp = visual_map.get(section["heading"])
    label = visual_type_label(section.get("visual_type"))
    prompt_text = vp["image_prompt"] if vp and vp.get("image_prompt") else "(no prompt generated)"
    return f'> 🖼️ **[Image placeholder — {label}]**\n> "{prompt_text}"'

# ---------------------------------------------------------------------------
# Trending-topic helpers
# ---------------------------------------------------------------------------
def topic_to_api_payload(topic: dict) -> str:
    """The string passed to keyword_graph_app's `trending_search` input and
    to gather_links' `query` param. For a topic picked from the trending
    list, this is the ENTIRE opportunity block (JSON-stringified) so the
    downstream graphs have full context (business problem, scores, buyer
    stage, etc). For a manually typed topic, it's just that text."""
    if topic["source"] == "trending":
        return json.dumps(topic["raw_block"], ensure_ascii=False)
    return topic["label"]

def make_trending_topic(opportunity: dict) -> dict:
    return {
        "source": "trending",
        "label": opportunity.get("normalized_topic") or opportunity.get("keyword", ""),
        "raw_block": opportunity,
    }

def make_custom_topic(text: str) -> dict:
    return {
        "source": "custom",
        "label": text.strip(),
        "raw_block": None,
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def blog_json_to_markdown(blog: dict, visual_prompts: list = None) -> str:
    """Reassemble the structured JSON blog into a Markdown document, with an
    image placeholder (quoting the generated prompt) inserted right after
    the heading of any section that was flagged as needing a visual."""
    visual_map = build_visual_prompt_map(visual_prompts)
    lines = [f"# {blog['title']}", ""]
    lines.append(f"_{blog['meta_description']}_")
    lines.append("")
    if blog.get("slug"):
        lines.append(f"**Slug:** /{blog['slug']}  ")
    lines.append(f"**Tags:** {', '.join(blog['tags'])}  ")
    lines.append(f"**Estimated read time:** {blog['estimated_read_time_minutes']} min")
    lines.append("")
    for section in blog["sections"]:
        lines.append(f"## {section['heading']}")
        lines.append("")
        if section.get("needs_visual"):
            lines.append(image_placeholder_markdown(section, visual_map))
            lines.append("")
        lines.append(section["content"])
        lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(blog["conclusion"])
    return "\n".join(lines)

