import json
import re
import os
from dotenv import load_dotenv
load_dotenv()
import atexit
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Qdrant Cloud Configuration
QDRANT_URL = "https://138d05d6-0fcc-42d5-983f-41385270168a.sa-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "properties"

# Initialize Client & Model
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
atexit.register(client.close)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Semantic Mappings for specific intents
SEMANTIC_MAP = {
    "open_spaces": ["park", "garden", "green", "landscape", "yoga", "forest", "open", "outdoor"],
    "luxury": ["luxury", "premium", "exclusive", "high-end", "elite"],
}

def hard_filter(payload, bhk=None, max_price=None, locality=None):
    """
    Deterministically exclude invalid results.
    """
    p_bhks = payload.get("bhk", [])
    p_price_min = payload.get("price_min")
    p_locality = f"{payload.get('city', '')} {payload.get('locality', '')}".lower()

    # Normalization for common city names
    city_map = {"bangalore": "bengaluru", "bengaluru": "bangalore"}
    
    # 1. BHK Constraint (Strict)
    if bhk:
        try:
            target_bhk = float(bhk)
            # Ensure strict matching in the list of floats
            if target_bhk not in [float(x) for x in p_bhks]:
                return False
        except ValueError:
            pass

    # 2. Price Constraint (Strict - EXCLUSIVE as per user request)
    if max_price:
        try:
            limit = float(max_price)
            # User sample: "under 300" returning 300 -> ❌ exclude. 
            # Means: price_min must be strictly less than limit.
            if p_price_min is None or p_price_min >= limit:
                return False
        except ValueError:
            pass

    # 3. Locality Constraint (Strict check with normalization)
    if locality:
        target_loc = str(locality).lower()
        # Accept if direct match or known city mapping or if it's in the name or description
        searchable_text = f"{p_locality} {payload.get('name', '')} {payload.get('description', '')}".lower()
        if target_loc not in searchable_text and city_map.get(target_loc, "") not in searchable_text:
            return False

    return True

def intent_score(payload, intents):
    """
    Boost scores for projects that match semantic intents.
    """
    score = 0
    p_amenities = [a.lower() for a in payload.get("amenities", [])]
    p_desc = payload.get("description", "").lower()
    is_lux = payload.get("is_luxury", False)

    if not intents:
        return 0

    for intent in intents:
        # Luxury Boost (Deterministic)
        if intent == "luxury" and is_lux:
            score += 2.0
            
        # Semantic Amenity Boost
        keywords = SEMANTIC_MAP.get(intent, [])
        for k in keywords:
            if any(k in a for a in p_amenities) or k in p_desc:
                score += 1.0
                break 
    return score

def rerank(results, params):
    """
    Strict Filtering + Semantic Boosting + Ranking.
    """
    bhk = params.get("bhk")
    max_price = params.get("max_price")
    locality = params.get("locality")
    intents = params.get("intents", [])

    valid_results = []
    for r in results:
        payload = r.payload
        # 1. Apply HARD Filters
        if hard_filter(payload, bhk=bhk, max_price=max_price, locality=locality):
            # 2. Apply Intent Boosting (Semantic Layer)
            s = intent_score(payload, intents)
            final_score = r.score + s
            valid_results.append((final_score, payload))

    # Sort valid results by score
    valid_results.sort(key=lambda x: x[0], reverse=True)
    
    # Return exactly the Top 3
    return [p for _, p in valid_results[:3]]

def search_tool(query, params=None):
    """
    Strict retrieval pipeline with intent boosting.
    """
    if params is None:
        params = {}
        
    query_vector = model.encode(query).tolist()

    # Semantic Search (Broad Candidates)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=20
    ).points

    # Deterministic Filtering & Reranking
    return rerank(results, params)

# Compatibility wrappers
def search_projects(query: str):
    return search_tool(query)

def filter_projects(bhk=None, location=None):
    return search_tool("", params={"bhk": bhk, "locality": location})