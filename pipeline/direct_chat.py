from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from pipeline.llm_load import llm

def direct_chat_llm():
    qa_prompt=ChatPromptTemplate.from_messages([
        ('system',('you are a helpful and creative assistant. PLease answer to user queries based on the context provided without making it too long but creative and not cringe . Remember dont hallucinate -Refer context : {context}')),
        MessagesPlaceholder('chat_history'),
        ('human','{input}')
    ])
    chain=qa_prompt| llm()
    return chain