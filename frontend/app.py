"""FrameworkRAG — Streamlit frontend"""
import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="FrameworkRAG",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 FrameworkRAG")
st.markdown("Query LangChain, LlamaIndex, and Haystack docs via RAG, Graph, or Multi-Agent.")

ENDPOINTS = {
    "🔄 RAG (Vector Search)": "/query",
    "🔗 Graph RAG (Neo4j)": "/query/graph",
    "🤖 Multi-Agent (LangGraph)": "/query/agents",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    endpoint_label = st.selectbox(
        "Query mode",
        list(ENDPOINTS.keys()),
        index=0,
    )
    endpoint = ENDPOINTS[endpoint_label]
    top_k = st.slider("Result count", 1, 10, 5)
    st.divider()
    st.markdown("**Frameworks indexed:**")
    st.markdown("- LangChain (12,167 chunks)")
    st.markdown("- LlamaIndex (575 chunks)")
    st.markdown("- Haystack (754 chunks)")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['url']}]({s['url']}) — {s.get('section', '')}")
        if msg.get("mode"):
            st.caption(f"Mode: {msg['mode']}")

if prompt := st.chat_input("Ask about LangChain, LlamaIndex, or Haystack..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Querying {endpoint_label}..."):
            try:
                resp = requests.post(
                    f"{API_BASE}{endpoint}",
                    json={"query": prompt, "top_k": top_k},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"Sources ({len(sources)})"):
                            for s in sources:
                                url = s.get("url", "")
                                section = s.get("section", "")
                                content = s.get("content", "")[:200]
                                st.markdown(f"- **[{url}]({url})** — {section}")
                                st.markdown(f"  > {content}")
                    st.caption(f"Mode: {endpoint_label}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "mode": endpoint_label,
                    })
                else:
                    err = resp.text[:500]
                    st.error(f"Error {resp.status_code}: {err}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at {API_BASE}. Is the server running?")
            except Exception as e:
                st.error(f"Error: {e}")
