import json
import re
import os
from dotenv import load_dotenv
load_dotenv(override=True)
import atexit
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz

# Qdrant Cloud Configuration
QDRANT_URL = "https://138d05d6-0fcc-42d5-983f-41385270168a.sa-east-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "properties"

# Lazy initializers for standalone use
_client = None
_model = None

def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        atexit.register(_client.close)
    return _client

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# Semantic Mappings for specific intents
SEMANTIC_MAP = {
    "open_spaces": ["park", "garden", "green", "landscape", "yoga", "forest", "open", "outdoor"],
    "luxury": ["luxury", "premium", "exclusive", "high-end", "elite"],
}

def hard_filter(payload, bhk=None, max_price=None, locality=None):
    """
    Deterministically exclude invalid results.
    """
    p_name = payload.get("name", "Unknown")
    p_bhks = payload.get("bhk", [])
    p_price_min = payload.get("price_min")
    p_locality_raw = f"{payload.get('city', '')} {payload.get('locality', '')} {payload.get('name', '')} {payload.get('description', '')}".lower()

    # 1. BHK Constraint (Strict)
    if bhk:
        try:
            target_bhk = float(bhk)
            # Ensure strict matching in the list of floats
            if target_bhk not in [float(x) for x in p_bhks]:
                print(f"DEBUG: {p_name} excluded - BHK {p_bhks} does not match {target_bhk}")
                return False
        except (ValueError, TypeError):
            pass

    # 2. Price Constraint (Soft - 10% Buffer)
    if max_price:
        try:
            limit = float(max_price)
            # Add 10% buffer as per request (inclusive match)
            buffer_limit = limit * 1.1
            if p_price_min is not None and p_price_min > buffer_limit:
                print(f"DEBUG: {p_name} excluded - Price {p_price_min} > Limit {buffer_limit} (incl. 10% buffer)")
                return False
        except (ValueError, TypeError):
            pass

    # 3. Locality Constraint (Sequential Word Boundary)
    if locality:
        target_loc = str(locality).lower()
        # Common variations/abbreviations
        location_mapping = {
            "omr": "old madras road",
            "orr": "outer ring road",
            "sarjapur": "sarjapur road",
            "blore": "bengaluru",
            "blr": "bengaluru",
            "bangalore": "bengaluru",
            "hyd": "hyderabad",
            "chennai": "chennai"
        }
        target_loc = location_mapping.get(target_loc, target_loc)
        
        # Proximity Locality matching (Allows up to 3 filler words between tokens)
        tokens = [re.escape(t) for t in re.split(r'[^a-zA-Z0-9]+', target_loc) if t]
        if tokens:
            separator = r"(?:\W+\w+){0,3}\W+"
            pattern = rf"\b{separator.join(tokens)}\b"
            if not re.search(pattern, p_locality_raw):
                print(f"DEBUG: {p_name} excluded - Locality '{target_loc}' not found within proximity")
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
    offset = params.get("offset", 0)
    sort_price_asc = params.get("sort_price_asc", False)

    valid_results = []
    for r in results:
        payload = r.payload
        # 1. Apply HARD Filters
        if hard_filter(payload, bhk=bhk, max_price=max_price, locality=locality):
            # 2. Apply Intent Boosting (Semantic Layer)
            s = intent_score(payload, intents)
            final_score = r.score + s
            valid_results.append((final_score, payload))

    # Add explicit price sorting if requested
    if sort_price_asc:
        print(f"DEBUG: Explicit price sorting enabled.")
        valid_results.sort(key=lambda x: x[1].get("price_min") if x[1].get("price_min") is not None else float('inf'))
    else:
        # Sort valid results by score (semantic + intent)
        valid_results.sort(key=lambda x: x[0], reverse=True)
    
    # Return exactly the Top 3 starting from the given offset
    return [p for _, p in valid_results[offset:offset+3]]

def search_tool(query, params=None, client=None, model=None):
    """
    Strict retrieval pipeline with intent boosting.
    """
    if params is None:
        params = {}
    
    # Use provided resources or fallback to lazy defaults
    active_client = client or get_client()
    active_model = model or get_model()
        
    query_vector = active_model.encode(query).tolist()

    # Semantic Search (Broad Candidates)
    results = active_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=50
    ).points

    # Deterministic Filtering & Reranking
    return rerank(results, params)

# Compatibility wrappers
def search_projects(query: str):
    return search_tool(query)

def filter_projects(bhk=None, location=None, sort_price_asc=None):
    return search_tool("", params={"bhk": bhk, "locality": location, "sort_price_asc": sort_price_asc})