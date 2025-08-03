import streamlit as st
from rag_agent import initialize_rag

# Initialize the RetrievalQA chain
qa_chain = initialize_rag()

# Streamlit UI
st.set_page_config(page_title="College Admission Agent", layout="centered")
st.title("🎓 College Admission Agent (RAG-Based)")
st.markdown("Ask me anything from the **College Prospectus 2025-2026**!")

# Input box
question = st.text_input("📌 Your question:")

# Answer box
if question:
    with st.spinner("🔍 Searching..."):
        answer = qa_chain.run(question)
    st.success("✅ Answer:")
    st.write(answer)
