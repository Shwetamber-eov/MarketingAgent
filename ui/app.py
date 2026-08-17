"""
Streamlit dashboard: enter a keyword, generate a professional blog (as
structured JSON) via LangChain + LangGraph + Gemini, and view/download
the result.

Run with:
    streamlit run app.py
"""

import os
import json
import datetime

import streamlit as st
from dotenv import load_dotenv

from MarketingAgent.backend.graph import generate_blog

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

    env_key_present = bool(os.environ.get("GOOGLE_API_KEY"))
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
        value=os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash-lite"),
        help="Must match a model string your API key can access.",
    )
    os.environ["GOOGLE_MODEL"] = model_name

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
    st.caption("Pipeline: Plan → Draft → Polish (LangGraph)")
    st.caption("Every step returns structured JSON, not free text.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def blog_json_to_markdown(blog: dict) -> str:
    """Reassemble the structured JSON blog into a Markdown document."""
    lines = [f"# {blog['title']}", ""]
    lines.append(f"_{blog['meta_description']}_")
    lines.append("")
    lines.append(f"**Tags:** {', '.join(blog['tags'])}  ")
    lines.append(f"**Estimated read time:** {blog['estimated_read_time_minutes']} min")
    lines.append("")
    for section in blog["sections"]:
        lines.append(f"## {section['heading']}")
        lines.append("")
        lines.append(section["content"])
        lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(blog["conclusion"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📝 AI Blog Generator")
st.caption("Powered by LangChain + LangGraph + Gemini · structured JSON output")

if "history" not in st.session_state:
    st.session_state.history = []

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
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("Please provide a Google API key in the sidebar.")
    else:
        with st.status("Generating your blog post...", expanded=True) as status:
            try:
                st.write("🧠 Planning (title, outline, tags) — JSON...")
                result = generate_blog(
                    keyword=keyword.strip(),
                    tone=tone,
                    audience=audience,
                    length=length,
                )
                st.write("✍️ Drafting sections — JSON...")
                st.write("🪄 Polishing final structured blog — JSON...")
                status.update(label="Blog generated!", state="complete")

                final_blog = result.get("final_blog")
                if not final_blog:
                    raise ValueError("No structured blog was returned by the pipeline.")

                st.session_state.history.insert(
                    0,
                    {
                        "keyword": keyword.strip(),
                        "blog": final_blog,  # structured dict
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    },
                )
            except Exception as e:
                status.update(label="Generation failed", state="error")
                st.exception(e)

# ---------------------------------------------------------------------------
# Display latest / history
# ---------------------------------------------------------------------------
if st.session_state.history:
    latest = st.session_state.history[0]
    blog = latest["blog"]

    st.divider()
    st.subheader(blog["title"])
    st.caption(
        f"Keyword: {latest['keyword']} · Generated {latest['timestamp']} · "
        f"~{blog['estimated_read_time_minutes']} min read"
    )
    st.caption(blog["meta_description"])
    st.write(" ".join(f"`{tag}`" for tag in blog["tags"]))

    tab_rendered, tab_json = st.tabs(["📖 Rendered", "🧾 Raw JSON"])

    with tab_rendered:
        for section in blog["sections"]:
            st.markdown(f"### {section['heading']}")
            st.markdown(section["content"])
        st.markdown("### Conclusion")
        st.markdown(blog["conclusion"])

    with tab_json:
        st.json(blog)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(blog, indent=2),
            file_name=f"{blog['title'].replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "⬇️ Download Markdown",
            data=blog_json_to_markdown(blog),
            file_name=f"{blog['title'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if len(st.session_state.history) > 1:
        st.divider()
        with st.expander(f"🕘 Previous generations ({len(st.session_state.history) - 1})"):
            for item in st.session_state.history[1:]:
                b = item["blog"]
                st.markdown(f"**{b['title']}** — _{item['timestamp']}_")
                st.caption(b["meta_description"])
                st.divider()
else:
    st.info("Enter a keyword above and click **Generate Blog** to get started.")