"""
Streamlit UI for the multi-agent research system.

Run with:
    streamlit run app.py

Place this file next to pipeline.py, agents.py and tools.py.
It calls the same agents/chains that pipeline.py uses, but step by step,
so the UI can show live progress for each stage.
"""

import time
import traceback

import streamlit as st

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .step-caption { color: #888; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "topic_done" not in st.session_state:
    st.session_state.topic_done = ""
if "timings" not in st.session_state:
    st.session_state.timings = {}
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""


def set_example(text: str):
    st.session_state.topic_input = text


# ----------------------------------------------------------------------------
# Pipeline (same logic as pipeline.py, split so the UI can show progress)
# ----------------------------------------------------------------------------
def run_pipeline_with_ui(topic: str) -> dict:
    state = {}
    timings = {}

    # Step 1 - Search agent
    with st.status("Step 1 · Search agent is working...", expanded=True) as status:
        t0 = time.time()
        search_agent = build_search_agent()
        search_result = search_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Find recent, reliable and detailed information about: {topic}",
                    )
                ]
            }
        )
        state["search_results"] = search_result["messages"][-1].content
        timings["Search"] = time.time() - t0
        st.write("Search results collected.")
        status.update(label="Step 1 · Search complete ✅", state="complete", expanded=False)

    # Step 2 - Reader agent
    with st.status("Step 2 · Reader agent is scraping top resources...", expanded=True) as status:
        t0 = time.time()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:800]}",
                    )
                ]
            }
        )
        state["scraped_content"] = reader_result["messages"][-1].content
        timings["Reader"] = time.time() - t0
        st.write("Content scraped.")
        status.update(label="Step 2 · Scraping complete ✅", state="complete", expanded=False)

    # Step 3 - Writer chain
    with st.status("Step 3 · Writer is drafting the report...", expanded=True) as status:
        t0 = time.time()
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke(
            {"topic": topic, "research": research_combined}
        )
        timings["Writer"] = time.time() - t0
        st.write("Report drafted.")
        status.update(label="Step 3 · Report drafted ✅", state="complete", expanded=False)

    # Step 4 - Critic chain
    with st.status("Step 4 · Critic is reviewing the report...", expanded=True) as status:
        t0 = time.time()
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        timings["Critic"] = time.time() - t0
        st.write("Review finished.")
        status.update(label="Step 4 · Review complete ✅", state="complete", expanded=False)

    st.session_state.timings = timings
    return state


def to_text(value) -> str:
    """Chains may return a string or a message object; normalise to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return getattr(value, "content", str(value))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Research Settings")
    st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Latest advances in solid-state batteries",
    )

    run_clicked = st.button("🚀 Run research", type="primary", use_container_width=True)

    st.markdown("**Try an example**")
    examples = [
        "Latest advances in solid-state batteries",
        "State of AI agents in 2026",
        "Impact of microplastics on human health",
    ]
    for ex in examples:
        st.button(ex, key=f"ex_{ex}", on_click=set_example, args=(ex,), use_container_width=True)

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. 🔎 **Search agent** finds sources\n"
        "2. 📄 **Reader agent** scrapes the best one\n"
        "3. ✍️ **Writer** drafts the report\n"
        "4. 🧐 **Critic** reviews it"
    )

    if st.session_state.result:
        st.divider()
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.result = None
            st.session_state.topic_done = ""
            st.session_state.timings = {}
            st.rerun()

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
st.title("🔎 Multi-Agent Research Assistant")
st.caption("Search → Read → Write → Critique, all in one place.")

if run_clicked:
    topic = st.session_state.topic_input.strip()
    if not topic:
        st.warning("Please enter a research topic in the sidebar first.")
    else:
        try:
            st.session_state.result = run_pipeline_with_ui(topic)
            st.session_state.topic_done = topic
            st.toast("Research complete!", icon="✅")
        except Exception as e:
            st.session_state.result = None
            st.error(f"Something went wrong while running the pipeline: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

result = st.session_state.result

if result:
    st.subheader(f"Results for: {st.session_state.topic_done}")

    # Timing metrics
    timings = st.session_state.timings
    if timings:
        cols = st.columns(len(timings) + 1)
        for col, (name, secs) in zip(cols, timings.items()):
            col.metric(name, f"{secs:.1f}s")
        cols[-1].metric("Total", f"{sum(timings.values()):.1f}s")

    report_text = to_text(result.get("report"))
    feedback_text = to_text(result.get("feedback"))

    tab_report, tab_feedback, tab_search, tab_scrape = st.tabs(
        ["📝 Final Report", "🧐 Critic Feedback", "🔎 Search Results", "📄 Scraped Content"]
    )

    with tab_report:
        st.markdown(report_text)
        st.download_button(
            "⬇️ Download report (.md)",
            data=report_text,
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(feedback_text)

    with tab_search:
        st.markdown(to_text(result.get("search_results")))

    with tab_scrape:
        st.markdown(to_text(result.get("scraped_content")))
else:
    st.info("Enter a topic in the sidebar and click **Run research** to get started.")