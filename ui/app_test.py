import os
import json
import datetime
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# from backend.graph import generate_blog
from backend.graph_new import generate_blog
from backend.tools import isblogexist, send_blog_email
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
        value=os.getenv("GEMMA_MODEL", "gemma-4-31b-it"),
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
    st.caption("Pipeline: Plan → Draft → Visual Prompts → Polish → SEO Check (LangGraph)")
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


def run_generation(target_keyword: str):
    """Run the Plan → Draft → Visual Prompts → Polish → SEO Check pipeline
    for `target_keyword` and push the result onto history. Shared by both
    the fresh-keyword path and the merged-keyword path (used after a
    duplicate is confirmed by the user)."""
    with st.status("Generating your blog post...", expanded=True) as status:
        try:
            st.write("🧠 Planning (title, meta, slug, outline, tags) — JSON...")
            result = generate_blog(
                keyword=target_keyword,
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
                    "keyword": target_keyword,
                    "blog": final_blog,           # structured dict
                    "visual_prompts": visual_prompts,  # [{heading, visual_type, image_prompt}, ...]
                    "seo_report": seo_report,      # {score, checks, failed_checks, ...}
                    "seo_fix_attempts": fix_attempts,
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

# Flow state for the duplicate-keyword handshake:
#   "idle"                  -> normal state, nothing pending
#   "duplicate_found"       -> a similar blog exists, waiting for yes/no
#   "awaiting_merge_keyword"-> user said yes, waiting for a second keyword
if "flow_state" not in st.session_state:
    st.session_state.flow_state = "idle"
if "pending_keyword" not in st.session_state:
    st.session_state.pending_keyword = ""
if "duplicate_info" not in st.session_state:
    st.session_state.duplicate_info = None

col1, col2 = st.columns([4, 1])
with col1:
    keyword = st.text_input(
        "Enter a keyword or topic",
        placeholder="e.g. sustainable urban gardening",
    )
with col2:
    st.write("")
    st.write("")
    generate_clicked = st.button("Generate Blog 🚀", use_container_width=True, type="primary")

if generate_clicked:
    if not keyword.strip():
        st.error("Please enter a keyword first.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error("Please provide a Google API key in the sidebar.")
    else:
        with st.spinner("Checking for existing similar blogs..."):
            exists, message, details = isblogexist(keyword.strip())

        if exists:
            # Don't generate yet — stash the duplicate info and ask the user
            # whether they want to continue by merging in another keyword.
            st.session_state.flow_state = "duplicate_found"
            st.session_state.pending_keyword = keyword.strip()
            st.session_state.duplicate_info = {"message": message, "details": details}
        else:
            st.session_state.flow_state = "idle"
            st.session_state.pending_keyword = ""
            st.session_state.duplicate_info = None
            run_generation(keyword.strip())

# ---------------------------------------------------------------------------
# Duplicate-keyword handshake UI
# ---------------------------------------------------------------------------
if st.session_state.flow_state == "duplicate_found":
    info = st.session_state.duplicate_info
    st.error(f"⚠️ A similar blog already exists for **{st.session_state.pending_keyword}**: {info['message']}")

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

    st.warning("Do you want to continue anyway? You can merge this keyword with another topic to make it more unique.")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Yes, add another keyword", use_container_width=True):
            st.session_state.flow_state = "awaiting_merge_keyword"
            st.rerun()
    with col_no:
        if st.button("❌ No, cancel", use_container_width=True):
            st.session_state.flow_state = "idle"
            st.session_state.pending_keyword = ""
            st.session_state.duplicate_info = None
            st.rerun()

elif st.session_state.flow_state == "awaiting_merge_keyword":
    st.info(f"Original keyword: **{st.session_state.pending_keyword}**")
    extra_keyword = st.text_input(
        "Enter another keyword/topic to combine with the original",
        placeholder="e.g. indoor vertical farming",
        key="extra_keyword_input",
    )

    col_merge, col_cancel = st.columns(2)
    with col_merge:
        merge_clicked = st.button("🔀 Merge & Generate Blog", use_container_width=True, type="primary")
    with col_cancel:
        cancel_clicked = st.button("Cancel", use_container_width=True)

    if cancel_clicked:
        st.session_state.flow_state = "idle"
        st.session_state.pending_keyword = ""
        st.session_state.duplicate_info = None
        st.rerun()

    if merge_clicked:
        if not extra_keyword.strip():
            st.error("Please enter an additional keyword before merging.")
        else:
            merged_keyword = f"{st.session_state.pending_keyword} {extra_keyword.strip()}"
            st.session_state.flow_state = "idle"
            st.session_state.pending_keyword = ""
            st.session_state.duplicate_info = None
            st.success(f"Generating a blog for the merged topic: **{merged_keyword}**")
            run_generation(merged_keyword)

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
    st.info("Enter a keyword above and click **Generate Blog** to get started.")