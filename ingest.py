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
    text = str(text)
    # 1. Expand range like "2-3"
    ranges = re.findall(r'(\d+)\s*[-]\s*(\d+)', text)
    range_bhk = []
    for start, end in ranges:
        for i in range(int(start), int(end) + 1):
            range_bhk.append(i)
    # 2. Extract all numbers
    all_nums = re.findall(r'\d+\.?\d*', text)
    found_bhk = [float(x) if "." in x else int(x) for x in all_nums]
    # Merge and unique
    return sorted(list(set(range_bhk + found_bhk)))

def extract_price_range(text):
    if not text:
        return None, None
    text = str(text).replace("*", "").replace(",", "").lower()
    
    # 1. Capture numbers WITH units (high priority)
    # Matches "8.40 Crore", "95 Lakhs", etc.
    unit_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(cr|crore|l|lakhs?)', text)
    nums = []
    if unit_matches:
        for val, unit in unit_matches:
            v = float(val)
            if any(k in unit for k in ["cr", "crore"]):
                v *= 100
            nums.append(v)
    
    # 2. Fallback to numbers without units (lower priority)
    if not nums:
        # Scrub BHK/Bed counts properly
        # Matches numbers followed by "&", "and", ",", "BHK", "Bed" etc.
        clean_text = re.sub(r'\d+(?=\s*(?:&|and|,|bhk|bed|bedroom|beds))', '', text)
        # Also catch the final one: "4 BHK"
        clean_text = re.sub(r'\d+\s*(?:bhk|bed|beds|bedroom|bedrooms)', '', clean_text)
        
        any_nums = re.findall(r'(\d+(?:\.\d+)?)', clean_text)
        for val in any_nums:
            v = float(val)
            if ("cr" in text or "crore" in text) and "lakh" not in text:
                v *= 100
            nums.append(v)

    if not nums:
        return None, None
    
    # Return as min, max in Lakhs
    if len(nums) == 1:
        if nums[0] >= 100000: # Handle raw rupees
            nums[0] /= 100000
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
    nearby_text = ""
    nearby = p.get("nearby", {})
    if isinstance(nearby, dict):
        for cat, places in nearby.items():
            for item in places:
                nearby_text += f"{item.get('place','')} {cat} "
    
    return f"{p.get('name','')} {p.get('description','')} {p.get('city','')} {p.get('locality','')} {p.get('property_type','')} {', '.join(p.get('amenities',[]))} {nearby_text}".strip()

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
        if not name or "detail page" in name.lower():
            skipped_junk += 1
            continue
            
        bhk = extract_bhk(p.get("type"))
        
        # Determine property type
        prop_type = p.get("property_type") 
        if not prop_type:
            if bhk:
                prop_type = "Apartment"
            elif any(k in (p.get("type") or "").lower() or k in (p.get("description") or "").lower() for k in ["plot", "land"]):
                prop_type = "Plot"
            else:
                prop_type = "Apartment"

        p_min, p_max = extract_price_range(p.get("price"))
        url = p.get("url", "")
        url_parts = url.split("/")
        inferred_city = url_parts[-2] if len(url_parts) > 2 else ""

        payload = {
            "name": name,
            "city": p.get("city", inferred_city).capitalize(),
            "locality": p.get("locality", "").capitalize(),
            "bhk": bhk,
            "property_type": prop_type,
            "price_min": p_min,
            "price_max": p_max,
            "amenities": p.get("amenities", []),
            "nearby": p.get("nearby", {}),
            "faqs": p.get("faqs", []),
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
