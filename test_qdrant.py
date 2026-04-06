import json
import re
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Credentials
QDRANT_URL = "https://138d05d6-0fcc-42d5-983f-41385270168a.sa-east-1-0.aws.cloud.qdrant.io:6333"
# Trying with the exact string provided, including the trailing > if that was intended
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
COLLECTION_NAME = "properties"

def run_ingestion():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Try a simple "whoami" or list_collections to verify auth
    try:
        collections = client.get_collections()
        print("Connected successfully!")
    except Exception as e:
        print(f"Auth failed: {e}")
        return

if __name__ == "__main__":
    run_ingestion()
