import streamlit as st
import nltk
import time
from langchain_core.messages import HumanMessage, AIMessage
from pipeline.session_history import generate_unique_sessionID
from pipeline.cache import cache_answer, get_cached_answer
from pipeline.rag_pipeline import rag_pipe
from pipeline.direct_chat import direct_chat_llm

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# --------------------- Init Session ---------------------
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


# --------------------- File Save ---------------------
def save_uploaded_files(files):
    new_files = []
    for f in files:
        if f.name not in st.session_state.uploaded:
            with open(f.name, "wb") as out:
                out.write(f.getbuffer())
            st.session_state.uploaded.append(f.name)
            new_files.append(f.name)
    return new_files


# --------------------- RAG Mode ---------------------
def rag_mode():
    st.title("📚 RAG Chatbot with Memory")
    st.write("Upload documents and chat with their content.")

    # File upload
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

            # Initialize RAG chain
            st.session_state.rag_chain, _ = rag_pipe(
                st.session_state.uploaded, st.session_state.rag_id
            )

            st.success("Documents processed! You can ask questions now.")
            st.session_state.processing = False

    # Show uploaded docs
    if st.session_state.uploaded:
        st.subheader("Uploaded Documents")
        for d in st.session_state.uploaded:
            st.write("- ", d)

    # Display chat history
    for msg in st.session_state.rag_history:
        with st.chat_message("assistant" if isinstance(msg, AIMessage) else "user"):
            st.markdown(msg.content)

    # User input
    question = st.chat_input("Ask about your documents...")

    if question:
        if st.session_state.processing:
            st.warning("Wait… still processing.")
            st.stop()

        if not st.session_state.uploaded:
            st.warning("Upload at least one document.")
            return

        # Add user message to history
        st.session_state.rag_history.append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.markdown(question)

        # Check cache
        cached = get_cached_answer(question, st.session_state.rag_id)
        if cached:
            answer = cached
        else:
            # ----------------- FIXED -----------------
            response = st.session_state.rag_chain({
                "question": question,
                "chat_history": st.session_state.rag_history
            })
            answer = response['answer']
            cache_answer(question, answer, st.session_state.rag_id)

        # Show assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Add assistant message to history
        st.session_state.rag_history.append(AIMessage(content=answer))# --------------------- Direct LLM Chat ---------------------
def direct_mode():
    st.title("🧠 Direct Chat Mode")

    # Display history
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


# --------------------- Main ---------------------
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
