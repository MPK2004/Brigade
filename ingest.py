import json
import re
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Cloud Configuration
QDRANT_URL = "https://138d05d6-0fcc-42d5-983f-41385270168a.sa-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "properties"

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data/brigade.json")
with open(DATA_PATH, "r") as f:
    DATA = json.load(f)

def extract_bhk(text):
    if not text:
        return []
    matches = re.findall(r'\d+\.?\d*', str(text))
    return [float(x) if "." in x else int(x) for x in matches]

def extract_price_range(text):
    if not text:
        return None, None
    text = str(text).lower()
    nums = re.findall(r'\d+\.?\d*', text)
    nums = [float(x) for x in nums if x != "."]
    if "cr" in text:
        nums = [x * 100 for x in nums]
    if len(nums) == 1:
        return nums[0], None
    elif len(nums) >= 2:
        return nums[0], nums[1]
    return None, None

def is_luxury(price_min, description):
    if price_min and price_min >= 200:
        return True
    if description:
        desc = description.lower()
        if any(k in desc for k in ["luxury", "premium", "exclusive"]):
            return True
    return False

def build_text(p):
    return f"{p.get('name','')} {p.get('description','')} {p.get('city','')} {p.get('locality','')} {', '.join(p.get('amenities',[]))}"

def run_ingestion():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Purging collection...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    valid_payloads = []
    skipped_junk = 0
    skipped_schema = 0

    print(f"Preprocessing data...")
    for p in DATA:
        name = p.get("name", "")
        if "detail page" in name.lower():
            skipped_junk += 1
            continue
            
        bhk = extract_bhk(p.get("type"))
        if not bhk or not name:
            skipped_schema += 1
            continue

        p_min, p_max = extract_price_range(p.get("price"))
        url = p.get("url", "")
        url_parts = url.split("/")
        inferred_city = url_parts[-2] if len(url_parts) > 2 else ""

        payload = {
            "name": name,
            "city": p.get("city", inferred_city).capitalize(),
            "locality": p.get("locality", "").capitalize(),
            "bhk": bhk,
            "price_min": p_min,
            "price_max": p_max,
            "amenities": p.get("amenities", []),
            "is_luxury": is_luxury(p_min, p.get("description")),
            "description": p.get("description"),
            "url": url,
        }
        valid_payloads.append(payload)

    print(f"Batch encoding {len(valid_payloads)} texts...")
    texts = [build_text(p) for p in valid_payloads]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()

    print(f"Upserting to Cloud...")
    points = [
        PointStruct(id=i, vector=vectors[i], payload=valid_payloads[i])
        for i in range(len(valid_payloads))
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    print(f"✅ Success! Ingested {len(points)} properties.")
    print(f"Skipped {skipped_junk} junk and {skipped_schema} schema violations.")

if __name__ == "__main__":
    run_ingestion()
