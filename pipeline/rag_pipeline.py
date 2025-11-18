from langchain.prompts import PromptTemplate
from langchain.chains import conversational_retrieval
from pipeline.embeddings import create_embeddings
from pipeline.session_history import generate_unique_sessionID
from pipeline.vector_store import vector_store_index
from pipeline.llm_load import llm
from utils.load_docs import load_user_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from pinecone_text.sparse import BM25Encoder

def rag_pipe(sources, session_id=None, index_name="project-2-pinecone"):
    if session_id is None:
        session_id = generate_unique_sessionID()
    namespace = session_id

    documents = load_user_documents(sources)
    if not documents:
        raise ValueError("Please provide documents to proceed.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    splitted_documents = splitter.split_documents(documents)

    texts = [doc.page_content for doc in splitted_documents]
    bm25_encoder = BM25Encoder()
    bm25_encoder_final = bm25_encoder.fit(texts)

    embeddings_model = create_embeddings()

    vector_store = Pinecone.from_documents(
        documents=splitted_documents,
        embedding=embeddings_model,
        index_name=index_name,
        namespace=namespace
    )

    hybrid_retriever = vector_store.as_retriever(
        hybrid_search=True,
        sparse_encoder=bm25_encoder_final,
        text_key="page_content",
        namespace=namespace
    )

    condense_prompt = PromptTemplate(
        template="Given the conversation so far and the question, rephrase the question if needed.\n\nQuestion: {question}\nRephrased Question:",
        input_variables=["question"]
    )

    qa_prompt = PromptTemplate(
        template="You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )

    conversational_rag_chain = conversational_retrieval.from_llm(
        llm=llm(),
        retriever=hybrid_retriever,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False
    )

    return conversational_rag_chain, session_id