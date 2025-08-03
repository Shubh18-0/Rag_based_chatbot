import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2']='true'
Pinecone_api_key=os.getenv('pinecone_api_key')
Groq_api_key=os.getenv('groq_api_key')
Huggingface_api_key=os.getenv('huggingface_api_key')
Langchain_api_key=os.getenv('langchain_api_key')