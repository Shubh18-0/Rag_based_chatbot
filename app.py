"""Main application file for chatbot. In this file we :
1. Start by importing from various util files we prepared ,
2. then importing necessary libraries to work with; and then,
3. Implementing the structure for ui where users can chat with out chatbot application via Streamlit,
4. The bot also provides user with different options whether to chat with documents vs without documents.
5. We have implemented multiple state sessions to work and store session histories and append messages throught the session of a user"""

import streamlit as st
import nltk
import time
from langchain_core.messages import HumanMessage, AIMessage
from pipeline.session_history import generate_unique_sessionID
from pipeline.cache import cache_answer, get_cached_answer
from pipeline.rag_pipeline import rag_pipe
from pipeline.direct_chat import direct_chat_llm

nltk.download('punkt', quiet=True)

# first we we will initialize sessson state variables for rag and direct chat 
def init_session():
    defaults = {
        'rag_id': generate_unique_sessionID(),
        'direct_id': generate_unique_sessionID(),
        'rag_history': [],
        'direct_history': [],
        'rag_chain': None,
        'direct_chain': None,
        'temperature': 0.7,
        'uploaded': [],
        'processing': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# saving uploaded files so that we dont recompute any file later on
def save_uploaded_files(files):
    new_files = []
    for f in files:
        if f.name not in st.session_state.uploaded:
            with open(f.name, 'wb') as file_out:
                file_out.write(f.getbuffer())
            st.session_state.uploaded.append(f.name)
            new_files.append(f.name)
    return new_files

# rag mode will be respinsible for taking in user documents and will handle all the process of chunking , embedding and retrieval as we are calling rag pipeline method inside the function
def rag_mode():
    st.title("RAG chatbot with memory 🧠")
    st.write("Upload documents (PDF, TXT, DOCX) and chat with their content.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    if uploaded_files:
        new_files = save_uploaded_files(uploaded_files)
        if new_files:
            st.session_state.processing = True
            with st.spinner("Processing documents..."):
                for step in ["Loading🔃", "Chunking🧩", "Embedding", "interacting with vector store..📳", "Building RAG🏭","One sec..🕤"]:
                    st.write(f"⏳ {step}..")
                    time.sleep(0.95)
            st.session_state.rag_chain, _ = rag_pipe(st.session_state.uploaded, st.session_state.rag_id)
            st.success("Documents processed! You can start chatting now.")
            st.session_state.processing = False

    if st.session_state.uploaded:
        st.subheader("Uploaded Documents")
        for doc in st.session_state.uploaded:
            st.write(f"- {doc}")

    for msg in st.session_state.rag_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    question = st.chat_input("Ask about your documents..")
    if question:
        if st.session_state.processing:
            st.warning("Still processing documents. Please wait.")
            st.stop()

        if not st.session_state.uploaded:
            st.warning("Please upload at least one document first.")
            return

        st.session_state.rag_history.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.markdown(question)

        cached_answer = get_cached_answer(question, st.session_state.rag_id)
        if cached_answer:
            answer = cached_answer
        else:
            res = st.session_state.rag_chain.invoke(
        {"input": question,
        "chat_history": st.session_state.rag_history,
        "context": "",
        "temperature": st.session_state.temperature},
        config={"configurable": {"session_id": st.session_state.rag_id}})
            answer = res['answer']
            cache_answer(question, answer, st.session_state.rag_id)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.rag_history.append(AIMessage(content=answer))

# direct mode chat - user can chat without any documents 
def direct_mode():
    st.title("🧠 Direct Chat")
    st.write("Chat without uploading documents.")

    for msg in st.session_state.direct_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    question = st.chat_input("Ask me anything...")
    if question:
        st.session_state.direct_history.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.markdown(question)

        if not st.session_state.direct_chain:
            st.session_state.direct_chain = direct_chat_llm()

        cached_answer = get_cached_answer(question, st.session_state.direct_id)
        if cached_answer:
            answer = cached_answer
        else:
            res = st.session_state.direct_chain.invoke({
                "input": question,
                "chat_history": st.session_state.direct_history,
                "context": "",
                "temperature": st.session_state.temperature,
            })
            answer = res.content
            cache_answer(question, answer, st.session_state.direct_id)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.direct_history.append(AIMessage(content=answer))

def main():
    # when main function gets executed - first thing would be to set the session state variables - init session does that for us
    init_session()

    with st.sidebar:
        st.title("📐Chatbot Settings")
        st.markdown("---")
        mode = st.radio("Choose Mode:", ["Rag(Docs)📚", "Direct Chat🧠"])
        st.session_state.temperature = st.slider(
            "Creativity (temperature)", 0.0, 1.0, st.session_state.temperature, 0.05)

    if mode == "Rag(Docs)📚":
        rag_mode()
    else:
        direct_mode()

# main execution of file 
if __name__ == "__main__":
    main()
