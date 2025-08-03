from pinecone import Pinecone, ServerlessSpec
from config.env_variables import Pinecone_api_key

pc = Pinecone(api_key=Pinecone_api_key)

def vector_store_index(index_name="project-2-pinecone"):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    index = pc.Index(index_name)
    return index