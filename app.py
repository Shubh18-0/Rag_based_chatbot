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
nltk.download('punkt_tab', quiet=True)

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
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def save_uploaded_files(files):
    new_files = []
    for f in files:
        if f.name not in st.session_state.uploaded:
            with open(f.name, "wb") as out:
                out.write(f.getbuffer())
            st.session_state.uploaded.append(f.name)
            new_files.append(f.name)
    return new_files

def rag_mode():
    st.title("📚 RAG Chatbot with Memory")
    st.write("Upload documents and chat with their content.")

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents", type=["pdf", "txt", "docx"], accept_multiple_files=True
    )

    if uploaded_files:
        new_files = save_uploaded_files(uploaded_files)

        if new_files:
            st.session_state.processing = True

            with st.spinner("Processing your documents…"):
                for step in [
                    "Loading🔃", "Chunking🧩", "Embedding",
                    "Vector Store📳", "Building RAG🏭", "Almost Done…🕤"
                ]:
                    st.write(f"⏳ {step}..")
                    time.sleep(0.75)

            st.session_state.rag_chain, _ = rag_pipe(
                st.session_state.uploaded, st.session_state.rag_id
            )

            st.success("Documents processed! You can ask questions now.")
            st.session_state.processing = False

    if st.session_state.uploaded:
        st.subheader("Uploaded Documents")
        for d in st.session_state.uploaded:
            st.write("- ", d)

    for msg in st.session_state.rag_history:
        with st.chat_message("assistant" if isinstance(msg, AIMessage) else "user"):
            st.markdown(msg.content)

    question = st.chat_input("Ask about your documents...")

    if question:
        if st.session_state.processing:
            st.warning("Wait… still processing.")
            st.stop()

        if not st.session_state.uploaded:
            st.warning("Upload at least one document.")
            return

        st.session_state.rag_history.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.markdown(question)

        cached = get_cached_answer(question, st.session_state.rag_id)
        if cached:
            answer = cached
        else:
            response = st.session_state.rag_chain({
                "question": question,
                "chat_history": st.session_state.rag_history
            })
            answer = response['answer']
            cache_answer(question, answer, st.session_state.rag_id)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.rag_history.append(AIMessage(content=answer))

def direct_mode():
    st.title("🧠 Direct Chat Mode")

    for msg in st.session_state.direct_history:
        with st.chat_message("assistant" if isinstance(msg, AIMessage) else "user"):
            st.markdown(msg.content)

    question = st.chat_input("Ask anything...")

    if question:
        st.session_state.direct_history.append(HumanMessage(content=question))

        with st.chat_message("user"):
            st.markdown(question)

        if not st.session_state.direct_chain:
            st.session_state.direct_chain = direct_chat_llm()

        cached = get_cached_answer(question, st.session_state.direct_id)
        if cached:
            answer = cached
        else:
            response = st.session_state.direct_chain.invoke({
                "input": question,
                "chat_history": st.session_state.direct_history,
                "temperature": st.session_state.temperature
            })
            answer = response.content
            cache_answer(question, answer, st.session_state.direct_id)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.direct_history.append(AIMessage(content=answer))

def main():
    init_session()

    with st.sidebar:
        st.title("⚙️ Settings")
        mode = st.radio("Choose Mode:", ["Rag(Docs)📚", "Direct Chat🧠"])
        st.session_state.temperature = st.slider(
            "Creativity (Temperature)",
            0.0, 1.0, st.session_state.temperature, 0.05
        )

    if mode == "Rag(Docs)📚":
        rag_mode()
    else:
        direct_mode()

if __name__ == "__main__":
    main()
