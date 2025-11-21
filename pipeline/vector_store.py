from pinecone import Pinecone as PineconeClient, ServerlessSpec
from config.env_variables import Pinecone_api_key

PINECONE_REGION = "us-east1"
pc = PineconeClient(api_key=Pinecone_api_key)

INDEX_NAME = "project-2-pinecone"
DIMENSION = 384

def vector_store_index():
    existing_indexes = pc.list_indexes()

    if INDEX_NAME not in existing_indexes.names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )

    return pc.Index(INDEX_NAME)