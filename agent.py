import os
from dotenv import load_dotenv
load_dotenv(override=True)
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from tools import search_tool

COMPANY_INFO = {
    "Residential Enquiries": "Toll Free: 1800 102 9977",
    "Customer Care": "1800 102 9480 / NRI: +91 96112 18222",
    "Email": "here4you@brigadegroup.com"
}

def get_supported_cities():
    """
    Dynamically extract unique cities from the dataset to define operating boundaries.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data/brigade.json")
    if not os.path.exists(data_path):
        return ["Bengaluru", "Chennai", "Hyderabad"]
    
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        cities = set()
        for p in data:
            name = p.get("name", "")
            if not name or "detail page" in name.lower():
                continue
            
            city = p.get("city")
            if not city:
                url = p.get("url")
                if url and isinstance(url, str):
                    url_parts = url.split("/")
                    city = url_parts[-2] if len(url_parts) > 2 else ""
            
            if city and isinstance(city, str):
                cleaned_city = city.replace("-", " ").title()
                cities.add(cleaned_city)
        return sorted(list(cities)) if cities else ["Bengaluru", "Chennai", "Hyderabad"]
    except Exception:
        return ["Bengaluru", "Chennai", "Hyderabad"]

SUPPORTED_CITIES = get_supported_cities()

class PlannerOutput(BaseModel):
    query: str = Field(description="The specific property name or rewritten query.")
    params: dict = Field(description="Dictionary containing bhk, max_price, locality, intents, offset, sort_price_asc.")
    is_property_search: bool = Field(description="True if it's a property search, False if chit-chat.")

# Split LLMs: 70b to think/plan reliably, 8b to respond rapidly
planner_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
responder_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

class AgentState(TypedDict):
    query: str
    history: List[dict] # Chat history for memory
    tool_args: dict
    tool_result: list
    response: str
    client: Optional[object] # QdrantClient
    model: Optional[object] # SentenceTransformer

def planner_node(state: AgentState):
    """
    Extracts search parameters using the latest query AND conversation history.
    """
    query = state["query"]
    history = state.get("history", [])
    
    # Format history for the prompt
    history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-5:]])
    
    system_prompt = f"""
    You are a precise real estate search specialist for Brigade Group.
    Your goal is to extract structured filters from the user's latest query, using conversation context.
    
    CONVERSATION HISTORY:
    {history_str}
    
    LATEST QUERY: {query}
    
    RULES:
    1. Reference Resolution: If the user says "the 2nd property", "the first one", or "it", replace the "query" field with the EXACT project name from the Assistant's previous message list.
    2. Filter Inheritance: Retain previous filters (like BHK or budget) ONLY IF the user hasn't changed them. If the user mentions a new location (e.g., 'what about whitefield'), UPDATE the locality parameter and OVERWRITE the old one.
    3. Pagination: If the user asks for "more", "others", or "next", preserve the previous filters but add an 'offset' parameter (integer, default 0) to skip previously shown properties (e.g., set offset to 3 to show the next 3).
    
    EXAMPLES:
    History:
    USER: 3 BHK in South Bangalore
    ASSISTANT: 1. Brigade Omega... 2. Brigade Panorama...
    LATEST QUERY: tell me about the second one
    OUTPUT: {{"query": "Brigade Panorama", "params": {{"bhk": 3, "locality": "South Bangalore"}}}}
    
    History:
    USER: 3 BHK in South Bangalore
    ASSISTANT: 1. Brigade Omega...
    LATEST QUERY: what about whitefield
    OUTPUT: {{"query": "whitefield", "params": {{"bhk": 3, "locality": "whitefield"}}}}

    History:
    USER: properties in bengaluru
    ASSISTANT: 1. Brigade Omega... 2. Brigade Panorama... 3. Brigade Symphony...
    LATEST QUERY: are there any others?
    OUTPUT: {{"query": "bengaluru", "params": {{"locality": "bengaluru", "offset": 3}}}}
    
    Return ONLY JSON with 'query', 'params' (bhk, max_price, locality, intents, offset, sort_price_asc), and 'is_property_search' (boolean).
    NOTE: 'locality' MUST capture any city, neighborhood, or area mentioned (e.g., 'Whitefield', 'Yelahanka', 'Chennai').
    NOTE: 'max_price' MUST be a number representing LAKHS (e.g., "1 Crore" = 100, "50 Lakhs" = 50).
    If the user asks for 'least', 'cheapest', or 'lowest price', set 'sort_price_asc': true in the params JSON.
    If the user's input is just a greeting (e.g., "hi", "hello") or general conversation (including asking for contact details, office locations, or support), set 'is_property_search' to false.
    """
    
    messages = [SystemMessage(content=system_prompt)]
    
    try:
        structured_llm = planner_llm.with_structured_output(PlannerOutput)
        response = structured_llm.invoke(messages)
        # Accommodate different Pydantic versions
        if hasattr(response, 'model_dump'):
            data = response.model_dump()
        else:
            data = response.dict()
            
        print(f"DEBUG_PLANNER: Successfully parsed JSON: {data}")
        return {"tool_args": data}
    except Exception as e:
        print(f"DEBUG_PLANNER Error: {e}")
        return {"tool_args": {"query": query, "params": {}}}

def tool_node(state: AgentState):
    """
    Deterministic search and filtering.
    """
    args = state["tool_args"]
    is_search = args.get("is_property_search", True)
    
    if not is_search:
        return {"tool_result": ["CHITCHAT"]}
        
    query = args.get("query", state["query"])
    params = args.get("params", {})
    
    # search_tool handles hard filters, intent boosts, and top 3
    # Inject resources from state if available
    result = search_tool(query, params, client=state.get("client"), model=state.get("model"))
    return {"tool_result": result}

def responder_node(state: AgentState):
    """
    Grounded conversational response.
    """
    result = state["tool_result"]
    query = state["query"]
    history = state.get("history", [])
    # Helper to convert history to explicit Langchain messages
    def get_conversation_messages(sys_prompt):
        msgs = [SystemMessage(content=sys_prompt)]
        for msg in history[-5:]:
            if msg["role"] == "user":
                msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                msgs.append(AIMessage(content=msg["content"]))
        msgs.append(HumanMessage(content=query))
        return msgs
    
    if result == ["CHITCHAT"]:
        system_prompt = f"""
        Act as a professional Property Advisor for Brigade Group.
        Respond warmly to the user's message.
        
        INTERNAL KNOWLEDGE:
        - COMPANY CONTACT INFO: {json.dumps(COMPANY_INFO, indent=2)}
        - SUPPORTED CITIES (SERVICE AREAS): {', '.join(SUPPORTED_CITIES)}
        
        RULES:
        1. SERVICE AREAS: If the user asks which cities/areas Brigade operates in, ONLY list: {', '.join(SUPPORTED_CITIES)}. Never invent or list cities not in this list.
        2. CONTACT INFO: If the user asks for contact details, phone numbers, or emails, provide the info from COMPANY CONTACT INFO.
        3. GREETINGS: If it's a greeting, respond warmly and ask how you can help them find a property today.
        4. PROACTIVE ENGAGEMENT: End the response with a single, helpful question to guide them toward finding their next home.
        5. Keep it brief and professional.
        """
        response = responder_llm.invoke(get_conversation_messages(system_prompt))
        return {"response": response.content}
    
    if not result:
        params = state.get("tool_args", {}).get("params", {})
        searched_loc = params.get("locality")
        
        if searched_loc and searched_loc.title() not in SUPPORTED_CITIES:
            return {"response": f"I see you are looking for properties in {searched_loc.title()}, but Brigade Group currently only operates in {', '.join(SUPPORTED_CITIES)}. Would you like to explore our projects in those cities instead?"}
        
        return {"response": f"No matching properties found under your specific criteria in our supported cities ({', '.join(SUPPORTED_CITIES)}). Would you like to check a different budget or location?"}

    # Identify if we're describing a single property (detail view) or listing many
    is_detail_view = len(result) == 1 or any(x in query.lower() for x in ["detail", "tell me about", "more about", "first", "second", "third", "2nd", "1st", "3rd", "it", "this"])

    # Deterministic Data Transformation: Handle all math and string formatting BEFORE LLM
    formatted_results = []
    for r in result:
        r_copy = dict(r)
        price = r_copy.get("price_min")
        if price is not None:
            if price >= 100:
                r_copy["formatted_price"] = f"₹{price/100:.2f} Cr"
            else:
                r_copy["formatted_price"] = f"₹{price} Lakhs"
        else:
            r_copy["formatted_price"] = "Price on Request"
        formatted_results.append(r_copy)

    if is_detail_view and len(formatted_results) > 0:
        # Provide detailed info for the top matched property
        system_prompt = f"""
        Act as a professional Property Advisor for Brigade Group.
        Provide a DETAILED, PREMIUM, AND CONVERSATIONAL response for this specific property.
        
        PROPERTY DATA:
        {json.dumps(formatted_results[0], indent=2)}
        
        GOLDEN RULES:
        1. GROUNDING: ONLY use the provided PROPERTY DATA to answer. Do NOT use your own knowledge about Bangalore, Chennai, or specific property locations. If 'nearby' or 'faqs' are empty or missing a specific category, say "I don't have that specific information for this property."
        2. NO HALLUCINATION: Do NOT invent nearby schools, hospitals, or landmarks. If the data says "Chrysalis High School (400 m)", use it. If it doesn't mention schools, do NOT list any.
        3. STRUCTURE: 
           - Use a warm, professional introduction.
           - Present 'Amenities' as a bulleted list.
           - Present 'Nearby Places' as a Markdown Table with columns: Category, Place, Distance.
        4. PRICING: Use 'formatted_price'. The data is already validated.
        5. CONTEXT: Maintain a helpful, advisory tone. Focus solely on this property.
        6. OPERATING BOUNDARIES: Brigade Group ONLY operates in {', '.join(SUPPORTED_CITIES)}. If the user asks about other cities, explicitly state we don't have properties there and offer to show projects in supported cities.
        7. PROACTIVE ENGAGEMENT: End the response with a compelling, open-ended question that anticipates the user's next need. For example, ask if they'd like to schedule a site visit, check specific unit availability, or see similar projects in this locality.
        """
        response = responder_llm.invoke(get_conversation_messages(system_prompt))
    else:
        system_prompt = f"""
        Act as a professional Property Advisor for Brigade Group.
        
        PROPERTIES (VALIDATED MATCHES):
        {json.dumps(formatted_results, indent=2)}
        
        RULES:
        1. HEADER: Start ONLY with the header exactly: "Top Matching Properties:". Do NOT include any introductory conversational text or greetings.
        2. LIST FORMAT: For each property, use the following Markdown structure:
           ### [name]
           - **Type:** [comma-separated list of BHKs] BHK (CRITICAL: OMIT this entire line if the 'bhk' list is empty)
           - **Starting Price:** [formatted_price]
           - **Highlights:** [locality] — [1-2 concise sentences from 'description' focusing on unique selling points]
        3. GROUNDING: Use ONLY the provided list. Do NOT suggest properties not in this list.
        4. PRICING: Use 'formatted_price'.
        5. ACCESSIBILITY: Use vertical stacking and bold labels. Absolutely NO pipes (|) or horizontal delimiters.
        6. OPERATING BOUNDARIES: Brigade Group ONLY operates in {', '.join(SUPPORTED_CITIES)}.
        7. PROACTIVE ENGAGEMENT: End the response with a single, helpful question to guide the user's next step. For example, ask if they want more details on a specific project or if they would like to adjust their budget or location filters.
        """
        response = responder_llm.invoke(get_conversation_messages(system_prompt))
    
    return {"response": response.content}

# Build Graph
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("tool", tool_node)
builder.add_node("responder", responder_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "tool")
builder.add_edge("tool", "responder")
builder.add_edge("responder", END)

graph = builder.compile()
