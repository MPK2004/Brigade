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
    If the user's input is just a greeting (e.g., "hi", "hello") or general conversation, set 'is_property_search' to false.
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
        system_prompt = """
        Act as a professional Property Advisor for Brigade Group.
        Respond warmly to the user's greeting or conversational message, and politely ask how you can assist them with finding a property today.
        Keep it brief, professional, and do not list any properties.
        """
        response = responder_llm.invoke(get_conversation_messages(system_prompt))
        return {"response": response.content}
    
    if not result:
        return {"response": "No matching properties found under your specific criteria. Would you like to check a different budget or location?"}

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
           - If there are 'FAQs', answer them naturally if relevant to the user's query.
        4. PRICING: Use 'formatted_price'. The data is already validated.
        5. CONTEXT: Maintain a helpful, advisory tone. Focus solely on this property.
        """
        response = responder_llm.invoke(get_conversation_messages(system_prompt))
    else:
        system_prompt = f"""
        Act as a professional Property Advisor for Brigade Group.
        
        PROPERTIES (VALIDATED MATCHES):
        {json.dumps(formatted_results, indent=2)}
        
        RULES:
        1. Start with the header exactly: "Top Matching Properties:"
        2. Format as a numbered list (1., 2., 3.): "Project Name — BHK | Price (use 'formatted_price') | Highlights"
        3. GROUNDING: Use ONLY the provided list. Do NOT suggest properties not in this list.
        4. PRICING: Use 'formatted_price'.
        5. TONALITY: Professional, concise, and helpful.
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
