"""Main application file for chatbot. In this file we :
1. Start by importing from various util files we prepared ,
2. then importing necessary libraries to work with; and then,
3. Implementing the structure for ui where users can chat with out chatbot application via Streamlit,
4. The bot also provides user with different options whether to chat with documents vs without documents.
5. We have implemented multiple state sessions to work and store session histories and append messages throught the session of a user"""

import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import time, uuid, os
from utils.load_docs import load_user_documents
from pipeline.session_history import generate_unique_sessionID
from pipeline.cache import cache_answer, get_cached_answer
from pipeline.direct_chat import direct_chat_llm
from pipeline.rag_pipeline import rag_pipe
from pipeline.vector_store import vector_store_index as vector_store
from langchain_core.messages import HumanMessage, AIMessage

# first we we will initialize sessson state variables for rag and direct chat 
if 'rag_session_id' not in st.session_state:
    st.session_state.rag_session_id = generate_unique_sessionID()
if 'direct_session_id' not in st.session_state:
    st.session_state.direct_session_id = generate_unique_sessionID()

if 'rag_history' not in st.session_state:
    st.session_state.rag_history = []
if 'direct_history' not in st.session_state:
    st.session_state.direct_history = []

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'uploaded_sources' not in st.session_state:
    st.session_state.uploaded_sources = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

#  Sidebar settings to choose from - Rag based mode - ie chat with documents or direct chat ie,- chat without docs
st.sidebar.title('📐Chatbot Settings')
mode = st.sidebar.radio('Choose mode:', ["Rag(Docs)📚", 'Direct Chat🧠'])

# Creativity button that controls how creative the replies from bot are
st.sidebar.markdown("🎨 Creativity")
st.session_state.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)

# If user mode selected = Rag mode - we will process docs for that specifically 
if mode == "Rag(Docs)📚":
    uploaded_files = st.sidebar.file_uploader('📂 Upload documents', accept_multiple_files=True,type=["pdf", "txt", "docx"])
    if uploaded_files:
        new_sources = []
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_sources:
                with open(file.name, 'wb') as f:
                    f.write(file.getbuffer())
                st.session_state.uploaded_sources.append(file.name)
                new_sources.append(file.name)
        # """
        # Once we have appended the sources in new source list thatwe created above - we will start processing documents with slight processing keywords passed in steps list anditerate for each step just to showcase progress of processing
        # """
        if new_sources:
            st.session_state.processing = True
            st.info("🔄 Processing documents...")

            progress_text = st.empty()
            progress_bar = st.progress(0)
            steps = ["Loading docs", "Chunking", "Generating embeddings", "Indexing in Vector store DB", "Building RAG" , "Almost done",'One sec..']
            for i, step in enumerate(steps):
                progress_text.text(f"⏳ {step}...")
                time.sleep(0.7)
                progress_bar.progress((i+1)/len(steps))

            st.session_state.rag_chain, _ = rag_pipe(
                st.session_state.uploaded_sources,
                st.session_state.rag_session_id
            )

            st.success("✅ All documents processed. You can now ask questions.")
            st.session_state.processing = False

# This would display our title for the bot with state managed temperature unless changed and the mode we are in at present
st.markdown("<h2 style='text-align:center;'>🤖 AI RAG Chatbot with Memory</h2>", unsafe_allow_html=True)
st.caption(f"Mode:{mode}| Temperature:{st.session_state.temperature}")

# For any of the selected mode - this block writes the content by AI and human both into the chat
if mode == 'Rag(Docs)📚':
    for msg in st.session_state.rag_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"🧑 **You:{msg.content}")
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(f"🤖 Bot:{msg.content}")
elif mode == 'Direct Chat🧠':
    for msg in st.session_state.direct_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"🧑 You: {msg.content}")
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(f"🤖Bot:{msg.content}")

# Once the user is done selecting the modes from above - He/she can now input their questions in the chat message bar 
user_input = st.chat_input("Type your question...")
if user_input:
    if st.session_state.processing:
        with st.chat_message("assistant"):
            st.write("🔄 I'm still processing your documents. Please wait a moment...")
        st.stop()

# # """ if the chosen mode was rag docs :
# #     then we will append the user message in rag history variable with human messsage and content
# #     else if there is no document uploaded by user we will add a catch to abide by it by displaying a provide doc statement
# """

    if mode == 'Rag(Docs)📚':
        st.session_state.rag_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(f"🧑 **You:** {user_input}")

        if not st.session_state.uploaded_sources:
            st.warning('⚠️ Please upload at least one document.')
            st.stop()

        # """In cache.py file we created a redis cache server so that user can get faster replies from cached data.
        # Below we are implementing a block to see if the results are cached we return them immediately to avoid useless computing and running the whole pipleine again
        # """
        try:
            cached = get_cached_answer(user_input, st.session_state.rag_session_id)
            if cached:
                with st.chat_message("assistant"):
                    st.markdown(f"🤖 **Bot:** {cached}")
                st.session_state.rag_history.append(AIMessage(content=cached))
            else:
                top_k = 3 if len(st.session_state.uploaded_sources) >= 3 else 5

                answer = st.session_state.rag_chain.invoke(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.rag_history,
                        "context": "",
                        "temperature": st.session_state.temperature,
                        "top_k": top_k
                    },
                    config={'configurable': {'session_id': st.session_state.rag_session_id}}
                )
                # """If the result was not alraedy cached- then we will use the function from cache.py to cache the answer for future use
                # """

                cache_answer(user_input, answer['answer'], st.session_state.rag_session_id)
                with st.chat_message("assistant"):
                    st.markdown(f"🤖 **Bot:** {answer['answer']}")
                st.session_state.rag_history.append(AIMessage(content=answer['answer']))
        except Exception as e:
            if "Sparse vector must contain at least one value" in str(e):
                with st.chat_message("assistant"):
                    st.write("⚠️ Your question was too short. Please be more specific.")
            else:
                raise ValueError
        # """IF the chosen mode was direct chat - we will same append the chat history separately  """
    elif mode == 'Direct Chat🧠':
        st.session_state.direct_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(f"🧑 **You:** {user_input}")

        if 'direct_chat_chain' not in st.session_state:
            st.session_state.direct_chat_chain = direct_chat_llm()

        cached = get_cached_answer(user_input, st.session_state.direct_session_id)
        if cached:
            with st.chat_message("assistant"):
                st.markdown(f"🤖 **Bot:** {cached}")
            st.session_state.direct_history.append(AIMessage(content=cached))
        else:
            answer = st.session_state.direct_chat_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.direct_history,
                "context": "",
                "temperature": st.session_state.temperature
            })
            cache_answer(user_input, answer.content, st.session_state.direct_session_id)
            with st.chat_message("assistant"):
                st.markdown(f"🤖 **Bot:** {answer.content}")
            st.session_state.direct_history.append(AIMessage(content=answer.content))
