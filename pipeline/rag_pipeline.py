from pipeline.embeddings import create_embeddings
from pipeline.session_history import create_session_history, generate_unique_sessionID
from pipeline.vector_store import vector_store_index
from pipeline.llm_load import llm
from utils.load_docs import load_user_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_community.vectorstores import Pinecone as Pineconevectorstore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore

def rag_pipe(sources, session_id=None, index_name="project-2-pinecone"):
    # ✅ Create unique namespace per session
    if session_id is None:
        session_id = generate_unique_sessionID()
    namespace = session_id

    documents = load_user_documents(sources)
    if not documents:
        raise ValueError('Error🔨!! Please provide documents to proceed :( )')

    model_instance = llm()

    # ✅ Split documents
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    splitted_documents = doc_splitter.split_documents(documents)
    texts = [doc.page_content for doc in splitted_documents]

    # ✅ Create BM25 sparse encoder
    bm25_encoder = BM25Encoder()
    bm25_encoder_final = bm25_encoder.fit(texts)

    # ✅ Store in Pinecone with namespace
    embeddings_model = create_embeddings()
    PineconeVectorStore.from_documents(
        documents=splitted_documents,
        embedding=embeddings_model,
        index_name=index_name,
        namespace=namespace
    )

    # ✅ Hybrid retriever restricted to namespace
    index = vector_store_index(index_name=index_name)
    hybrid_retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings_model,
        sparse_encoder=bm25_encoder_final,
        index=index,
        text_key='text',
        namespace=namespace
    )

    contextualized_q_prompt = ChatPromptTemplate.from_messages([
        ('system', 'Please rephrase user question into a standalone question using chat history :'),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    historyawareretriever = create_history_aware_retriever(
        llm=model_instance,
        prompt=contextualized_q_prompt,
        retriever=hybrid_retriever
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', "You are a helpful assistant! Please reply to user's query based on provided context: {context}"),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    document_chain = create_stuff_documents_chain(llm=model_instance, prompt=qa_prompt)
    retrieval_chain = create_retrieval_chain(historyawareretriever, document_chain)

    def get_session_history_for_chain(session_id: str):
        return create_session_history(session_id, model_instance.get_num_tokens)

    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history_for_chain,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    return conversational_rag_chain, session_id
