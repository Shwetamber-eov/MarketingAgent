import os
import json
import datetime
import concurrent.futures
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# from backend.graph import generate_blog
from backend.graph_new import generate_blog
from backend.graph_keywords import app as keyword_graph_app
from backend.tools import (
    isblogexist,
    send_blog_email,
    search_top_keywords,
    gather_links,
)
load_dotenv()

st.set_page_config(
    page_title="AI Blog Generator",
    page_icon="📝",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar - configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    env_key_present = bool(os.getenv("GOOGLE_API_KEY"))
    api_key_input = st.text_input(
        "Google API Key",
        value="",
        type="password",
        help="Leave blank to use GOOGLE_API_KEY from your .env file.",
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    if env_key_present or api_key_input:
        st.success("API key loaded ✅")
    else:
        st.warning("No API key found. Add one here or in a .env file.")

    model_name = st.text_input(
        "Gemini model name",
        # value=os.getenv("GEMMA_MODEL", "gemma-4-31b-it"),
        value=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        help="Must match a model string your API key can access.",
    )
    # os.environ["GOOGLE_MODEL"] = model_name

    st.divider()
    tone = st.selectbox(
        "Tone",
        ["professional", "conversational", "authoritative", "friendly", "technical"],
        index=0,
    )
    audience = st.text_input("Target audience", value="general readers")
    length = st.select_slider(
        "Length", options=["short", "medium", "long"], value="medium"
    )

    st.divider()
    st.caption("Pipeline: Trending topic → Keyword Strategy + Link Research (parallel) → Plan → Draft → Visual Prompts → Polish → SEO Check (LangGraph)")
    st.caption("Every step returns structured JSON, not free text.")
    st.caption("Sections flagged as needing a diagram show a placeholder with the generated image prompt, right where the image belongs.")

# NOTE: the sidebar above is commented out, but `tone`, `audience`, and
# `length` are still referenced later when calling generate_blog(). Defining
# them here keeps the app runnable until the sidebar is switched back on —
# swap these for the sidebar widgets whenever you re-enable that block.
# tone = "professional"
# audience = "general readers"
# length = "medium"

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


def render_image_placeholder(section: dict, visual_map: dict):
    """Streamlit widget shown right where an image/diagram belongs in a section."""
    vp = visual_map.get(section["heading"])
    icon = VISUAL_TYPE_ICONS.get(section.get("visual_type"), "🖼️")
    label = visual_type_label(section.get("visual_type"))
    with st.container(border=True):
        st.markdown(f"{icon} **Image placeholder — {label}** _(not yet generated)_")
        if vp and vp.get("image_prompt"):
            st.markdown(f'> "{vp["image_prompt"]}"')
        else:
            st.caption("No refined image prompt was generated for this section.")


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


def check_duplicate_and_advance(topic: dict):
    """Runs isblogexist() on `topic`'s label. If a duplicate is found, stash
    it and drop into the duplicate-resolution screen; otherwise move
    straight to generation. Shared by the initial topic confirmation and by
    every path that can produce a *new* topic afterwards (select different /
    add your own), so the duplicate check loops until the user explicitly
    continues anyway or cancels."""
    with st.spinner("Checking for existing similar blogs..."):
        exists, message, details = isblogexist(topic["label"])

    st.session_state.pending_topic = topic
    if exists:
        st.session_state.duplicate_info = {"message": message, "details": details}
        st.session_state.flow_state = "duplicate_found"
    else:
        st.session_state.duplicate_info = None
        st.session_state.flow_state = "generating"


def render_topic_selection():
    """Main topic-picking screen: choose one of the fetched trending
    opportunities, or type a topic of your own instead."""
    data = st.session_state.trending_data or {}
    summary = data.get("research_summary", {})
    opportunities = data.get("top_opportunities", [])

    if summary:
        with st.expander("📋 Research context", expanded=False):
            st.markdown(f"**Company:** {summary.get('company', '—')}")
            st.markdown(f"**Market:** {summary.get('market', '—')}")
            st.markdown(f"**Audiences:** {', '.join(summary.get('audiences', [])) or '—'}")
            st.markdown(f"**Research date:** {summary.get('research_date', '—')}")
            st.caption(summary.get("methodology_note", ""))

    st.subheader("🎯 Pick a trending topic")

    selected_idx = None
    if opportunities:
        selected_idx = st.selectbox(
            "Trending opportunities",
            options=list(range(len(opportunities))),
            format_func=lambda i: f"{opportunities[i].get('normalized_topic', 'Untitled topic')}  (rank {opportunities[i].get('rank', i + 1)})",
            key="topic_select_idx",
        )
        chosen = opportunities[selected_idx]
        st.caption(f"Keyword: {chosen.get('keyword', '—')}")
        with st.expander("More about this topic"):
            st.markdown(f"**Search intent:** {chosen.get('search_intent', '—')}")
            st.markdown(f"**Business problem:** {chosen.get('business_problem', '—')}")
            st.markdown(f"**Suggested angle:** {chosen.get('suggested_blog_angle', '—')}")
            st.markdown(f"**Overall priority score:** {chosen.get('overall_priority_score', '—')}")
    else:
        st.info("No trending opportunities were returned. Enter your own topic below.")

    st.markdown("**Or type your own topic instead:**")
    custom_text = st.text_input(
        "Your own topic",
        value="",
        key="custom_topic_input",
        placeholder="e.g. investment in FDE",
        label_visibility="collapsed",
    )

    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        confirm_clicked = st.button("✅ Confirm topic", use_container_width=True, type="primary")
    with col_cancel:
        cancel_clicked = st.button("Cancel", use_container_width=True)

    if confirm_clicked:
        topic = None
        if custom_text.strip():
            topic = make_custom_topic(custom_text)
        elif opportunities and selected_idx is not None:
            topic = make_trending_topic(opportunities[selected_idx])
        else:
            st.error("Pick a trending topic or type your own before confirming.")

        if topic is not None:
            check_duplicate_and_advance(topic)
            st.rerun()

    if cancel_clicked:
        st.session_state.flow_state = "idle"
        st.session_state.trending_data = None
        st.rerun()


def render_duplicate_screen():
    """Shown when isblogexist() flags the currently pending topic as a
    duplicate. Lets the user continue anyway, go back and pick a different
    trending topic, type a brand-new topic (fully replacing the flagged
    one), or cancel out of the flow entirely."""
    info = st.session_state.duplicate_info
    topic_label = st.session_state.pending_topic["label"]
    st.error(f"⚠️ A similar blog already exists for **{topic_label}**: {info['message']}")

    if info["details"]:
        verdict = info["details"]["verdict"]
        with st.expander("Why this was flagged as a duplicate", expanded=True):
            st.markdown(f"**Matched post:** {verdict.matched_title}")
            if verdict.matched_url:
                st.markdown(f"**URL:** {verdict.matched_url}")
            st.markdown(f"**Confidence:** {verdict.confidence:.0%}")
            st.markdown(f"**Reasoning:** {verdict.reasoning}")

            st.divider()
            st.caption("Other semantically similar posts considered:")
            for c in info["details"]["candidates"]:
                st.caption(f"- {c['title']} (distance={c['vector_distance']}) → {c['url']}")

    st.warning("What would you like to do?")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        continue_clicked = st.button("✅ Continue anyway", use_container_width=True, type="primary")
    with col_b:
        different_clicked = st.button("🔁 Different topic", use_container_width=True)
    with col_c:
        own_clicked = st.button("✏️ Add my own", use_container_width=True)
    with col_d:
        cancel_clicked = st.button("❌ Cancel", use_container_width=True)

    if continue_clicked:
        st.session_state.duplicate_info = None
        st.session_state.flow_state = "generating"
        st.rerun()

    if different_clicked:
        st.session_state.duplicate_info = None
        st.session_state.pending_topic = None
        st.session_state.flow_state = "topic_selection"
        st.rerun()

    if own_clicked:
        st.session_state.duplicate_info = None
        st.session_state.flow_state = "awaiting_replacement_topic"
        st.rerun()

    if cancel_clicked:
        st.session_state.flow_state = "idle"
        st.session_state.pending_topic = None
        st.session_state.duplicate_info = None
        st.session_state.trending_data = None
        st.rerun()


def render_replacement_topic_screen():
    """Shown after 'Add your own topic' from the duplicate screen. Typing a
    new topic here REPLACES the flagged one entirely (not merged with it)
    and is re-checked for duplicates, looping back to the duplicate screen
    again if it's also flagged."""
    st.info(f"Replacing: **{st.session_state.pending_topic['label']}**")
    new_text = st.text_input(
        "Enter a new topic to use instead",
        placeholder="e.g. AI-Native Product Engineering",
        key="replacement_topic_input",
    )
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        confirm_clicked = st.button("✅ Use this topic", use_container_width=True, type="primary")
    with col_cancel:
        cancel_clicked = st.button("Cancel", use_container_width=True)

    if cancel_clicked:
        st.session_state.flow_state = "idle"
        st.session_state.pending_topic = None
        st.session_state.duplicate_info = None
        st.session_state.trending_data = None
        st.rerun()

    if confirm_clicked:
        if not new_text.strip():
            st.error("Please enter a topic before continuing.")
        else:
            check_duplicate_and_advance(make_custom_topic(new_text))
            st.rerun()


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


def run_generation_pipeline(topic: dict):
    """Runs keyword_graph_app (keyword strategy) and gather_links (internal
    + external link research) in PARALLEL for `topic`, then feeds both
    results into generate_blog() together, running the
    Plan → Draft → Visual Prompts → Polish → SEO Check pipeline."""
    label = topic["label"]
    api_payload = topic_to_api_payload(topic)

    with st.status("Generating your blog post...", expanded=True) as status:
        try:
            st.write("🧭 Researching keyword strategy + reference links (in parallel)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                keyword_future = executor.submit(
                    keyword_graph_app.invoke, {"trending_search": api_payload}
                )
                links_future = executor.submit(gather_links, query=label)
                keyword_result = keyword_future.result()
                # links_result = links_future.result()[0].get("code")
                links_result = links_future.result()

            keyword_strategy = keyword_result["keyword_strategy"]
            internal_links = links_result.get("internal_links", [])
            external_links = links_result.get("external_links", [])

            st.write("🧠 Planning (title, meta, slug, outline, tags) — JSON...")
            result = generate_blog(
                keyword=label,
                keyword_strategy=keyword_strategy,
                internal_links=internal_links,
                external_links=external_links,
                tone=tone,
                audience=audience,
                length=length,
            )
            st.write("✍️ Drafting sections — JSON...")
            st.write("🖼️ Writing image/diagram prompts for flagged sections...")
            st.write("🪄 Polishing final structured blog — JSON...")

            final_blog = result.get("final_blog")
            if not final_blog:
                raise ValueError("No structured blog was returned by the pipeline.")

            visual_prompts = result.get("visual_prompts", [])
            seo_report = result.get("seo_report") or {}
            fix_attempts = result.get("seo_fix_attempts", 0)
            score = seo_report.get("score")

            if score is not None:
                st.write(f"🔍 SEO check — score {score}/100")
            if fix_attempts:
                st.write(f"🛠️ Auto-fixed SEO issues over {fix_attempts} round(s)")

            status.update(
                label=f"Blog generated! SEO score: {score}/100" if score is not None else "Blog generated!",
                state="complete",
            )

            st.session_state.history.insert(
                0,
                {
                    "keyword": label,
                    "blog": final_blog,           # structured dict
                    "visual_prompts": visual_prompts,  # [{heading, visual_type, image_prompt}, ...]
                    "seo_report": seo_report,      # {score, checks, failed_checks, ...}
                    "seo_fix_attempts": fix_attempts,
                    "internal_links": internal_links,
                    "external_links": external_links,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            )
        except Exception as e:
            status.update(label="Generation failed", state="error")
            st.exception(e)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📝 AI Blog Generator")
st.caption("Powered by LangChain + LangGraph + Gemini · structured JSON output")

if "history" not in st.session_state:
    st.session_state.history = []

# Flow state machine:
#   "idle"                       -> show "Get top searches" button
#   "topic_selection"            -> pick a trending topic, or type your own
#   "duplicate_found"            -> a similar blog exists, waiting for a choice
#   "awaiting_replacement_topic" -> user chose "add your own topic" from the
#                                    duplicate screen; typing a brand-new
#                                    topic that fully replaces the flagged one
#   "generating"                 -> parallel keyword_graph_app + gather_links,
#                                    then generate_blog
if "flow_state" not in st.session_state:
    st.session_state.flow_state = "idle"
if "trending_data" not in st.session_state:
    st.session_state.trending_data = None
if "pending_topic" not in st.session_state:
    st.session_state.pending_topic = None
if "duplicate_info" not in st.session_state:
    st.session_state.duplicate_info = None

if st.session_state.flow_state == "idle":
    fetch_clicked = st.button("🔎 Get top searches", type="primary")
    if fetch_clicked:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Please provide a Google API key in the sidebar.")
        else:
            with st.spinner("Fetching trending topics..."):
                trending_keywords=search_top_keywords()
                print(type(trending_keywords))
                print("================================\ntrending keywords: ",trending_keywords)
                st.session_state.trending_data = trending_keywords
            st.session_state.flow_state = "topic_selection"
            st.rerun()

elif st.session_state.flow_state == "topic_selection":
    render_topic_selection()

elif st.session_state.flow_state == "duplicate_found":
    render_duplicate_screen()

elif st.session_state.flow_state == "awaiting_replacement_topic":
    render_replacement_topic_screen()

elif st.session_state.flow_state == "generating":
    run_generation_pipeline(st.session_state.pending_topic)
    st.session_state.flow_state = "idle"
    st.session_state.pending_topic = None
    st.session_state.trending_data = None

# ---------------------------------------------------------------------------
# Display latest / history
# ---------------------------------------------------------------------------
if st.session_state.history:
    latest = st.session_state.history[0]
    blog = latest["blog"]
    visual_prompts = latest.get("visual_prompts", [])
    visual_map = build_visual_prompt_map(visual_prompts)
    seo_report = latest.get("seo_report", {})

    st.write("Send blog to user")
    if st.button("Send Email"):
        success = send_blog_email(
            blog_markdown=blog_json_to_markdown(blog, visual_prompts),
            title=blog.get("title")
        )

        print("::::::::::::::: success is ", success)
        if success:
            st.success("Email sent successfully!")
        else:
            st.error("Failed to send email.")

    #on_click only supports function not its return value
    print("after sending mail")
    st.divider()
    st.subheader(blog["title"])
    st.caption(
        f"Keyword: {latest['keyword']} · Generated {latest['timestamp']} · "
        f"~{blog['estimated_read_time_minutes']} min read"
    )
    if blog.get("slug"):
        st.caption(f"Slug: /{blog['slug']}")
    st.caption(blog["meta_description"])
    st.write(" ".join(f"`{tag}`" for tag in blog["tags"]))

    seo_score = seo_report.get("score")
    if seo_score is not None:
        score_icon = "🟢" if seo_score >= 80 else ("🟡" if seo_score >= 60 else "🔴")
        col_score, col_expander = st.columns([1, 3])
        with col_score:
            st.metric("SEO score", f"{seo_score}/100")
        with col_expander:
            with st.expander(f"{score_icon} SEO check breakdown", expanded=False):
                for name, check in seo_report.get("checks", {}).items():
                    check_icon = "✅" if check["passed"] else "❌"
                    line = f"{check_icon} **{name.replace('_', ' ').title()}**"
                    if not check["passed"]:
                        line += f" — {check['fix']}"
                    st.markdown(line)
                if latest.get("seo_fix_attempts"):
                    st.caption(f"Auto-fixed over {latest['seo_fix_attempts']} round(s).")

    tab_rendered, tab_visuals, tab_json = st.tabs(
        ["📖 Rendered", "🖼️ Visual Prompts", "🧾 Raw JSON"]
    )

    with tab_rendered:
        for section in blog["sections"]:
            st.markdown(f"### {section['heading']}")
            if section.get("needs_visual"):
                render_image_placeholder(section, visual_map)
            st.markdown(section["content"])
        st.markdown("### Conclusion")
        st.markdown(blog["conclusion"])

    with tab_visuals:
        if not visual_prompts:
            st.info("No sections in this post were flagged as needing a diagram or image.")
        else:
            st.caption("Copy any prompt below into an image-generation tool.")
            for vp in visual_prompts:
                icon = VISUAL_TYPE_ICONS.get(vp.get("visual_type"), "🖼️")
                st.markdown(f"{icon} **{vp['heading']}** — _{visual_type_label(vp.get('visual_type'))}_")
                st.code(vp["image_prompt"], language=None)
                st.divider()

    with tab_json:
        st.json({
            "blog": blog,
            "visual_prompts": visual_prompts,
            "seo_report": seo_report,
        })

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(
                {
                    "blog": blog,
                    "visual_prompts": visual_prompts,
                    "seo_report": seo_report,
                },
                indent=2,
            ),
            file_name=f"{blog['title'].replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "⬇️ Download Markdown",
            data=blog_json_to_markdown(blog, visual_prompts),
            file_name=f"{blog['title'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if len(st.session_state.history) > 1:
        st.divider()
        with st.expander(f"🕘 Previous generations ({len(st.session_state.history) - 1})"):
            for item in st.session_state.history[1:]:
                b = item["blog"]
                score = item.get("seo_report", {}).get("score")
                score_suffix = f" · SEO {score}/100" if score is not None else ""
                st.markdown(f"**{b['title']}** — _{item['timestamp']}_{score_suffix}")
                st.caption(b["meta_description"])
                st.divider()
else:
    st.info("Click **Get top searches** above to find a trending topic, or type your own once the list appears.")