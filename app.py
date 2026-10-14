import os
from importlib import import_module

import streamlit as st

try:
    os.environ.setdefault("OPENROUTER_API_KEY", st.secrets["OPENROUTER_API_KEY"])
    if "OPENROUTER_MODEL" in st.secrets:
        os.environ.setdefault("OPENROUTER_MODEL", st.secrets["OPENROUTER_MODEL"])
except (KeyError, FileNotFoundError):
    pass  

rag = import_module("08_prompting")

st.title("Course Support RAG Assistant")


question = st.text_area("Question")

if st.button("Answer") and question.strip():
    with st.spinner("Retrieving and answering..."):
        answer, sources = rag.answer_question(question)

    st.text_area("Answer", value=answer, height=220)

    with st.expander("Sources"):
        for source in sources:
            status = "CURRENT" if source["is_current"] else "OUTDATED"
            st.write(f"**{source['title']}** ({status}) - score {source['score']:.3f}")
            st.write(source["chunk_text"])
