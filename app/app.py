"""Streamlit UI for the Agentic RAG Chat Assistant (stretch goal).

Run:  streamlit run app/app.py
"""

import json
from pathlib import Path

import streamlit as st

from config.settings import TRACES_DIR

st.set_page_config(page_title="Agentic RAG Chat", layout="wide")


@st.cache_resource
def get_orchestrator():
    from orchestrator.orchestrator import ChatOrchestrator
    return ChatOrchestrator()


def main():
    st.title("Agentic RAG Chat Assistant")

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar: show latest trace
    with st.sidebar:
        st.header("Trace Viewer")
        trace_files = sorted(TRACES_DIR.glob("*_trace.json"), reverse=True)
        if trace_files:
            selected = st.selectbox(
                "Select trace",
                trace_files,
                format_func=lambda p: p.stem,
            )
            if selected:
                data = json.loads(selected.read_text(encoding="utf-8"))
                st.json(data)
        else:
            st.info("No traces yet. Send a message to generate one.")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask about HR policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        orch = get_orchestrator()
        with st.spinner("Thinking..."):
            answer = orch.handle_query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.rerun()


if __name__ == "__main__":
    main()
