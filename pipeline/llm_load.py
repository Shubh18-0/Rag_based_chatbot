from langchain_groq import ChatGroq
from config.env_variables import Groq_api_key

_model=None
def llm():
    global _model
    if _model is None:
        _model=ChatGroq(model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                        groq_api_key=Groq_api_key)
        
    return _model
    
