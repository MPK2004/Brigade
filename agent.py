import os
from dotenv import load_dotenv
load_dotenv(override=True)
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tools import search_tool

# Split LLMs: 70b to think/plan reliably, 8b to respond rapidly
planner_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)
responder_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

class AgentState(TypedDict):
    query: str
    history: List[dict] # Chat history for memory
    tool_args: dict
    tool_result: list
    response: str

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
    response = planner_llm.invoke(messages)
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        data = json.loads(content)
        print(f"DEBUG_PLANNER: {data}")
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
    result = search_tool(query, params)
    return {"tool_result": result}

def responder_node(state: AgentState):
    """
    Grounded conversational response.
    """
    result = state["tool_result"]
    query = state["query"]
    
    if result == ["CHITCHAT"]:
        system_prompt = f"""
        Act as a professional Property Advisor for Brigade Group.
        User Query: {query}
        
        Respond warmly to the user's greeting or conversational message, and politely ask how you can assist them with finding a property today.
        Keep it brief, professional, and do not list any properties.
        """
        messages = [SystemMessage(content=system_prompt)]
        response = responder_llm.invoke(messages)
        return {"response": response.content}
    
    if not result:
        return {"response": "No matching properties found under your specific criteria. Would you like to check a different budget or location?"}

    # Identify if we're describing a single property (detail view) or listing many
    # Usually, if "tell me about" or "details" or specific 1 property is found
    # We can detect if detail view based on result size and the user's intent to focus on one
    is_detail_view = len(result) == 1 or any(x in query.lower() for x in ["detail", "tell me about", "more about", "first", "second", "third", "2nd", "1st", "3rd", "it", "this"])

    if is_detail_view and len(result) > 0:
        # Provide detailed info for the top matched property
        system_prompt = f"""
        Act as a professional Property Advisor for Brigade Group.
        Provide a DETAILED AND CONVERSATIONAL response for this specific property.
        
        PROPERTY DATA:
        {json.dumps(result[0], indent=2)}
        
        RULES:
        1. Be professional, premium, and highly informative.
        2. Give a warm introduction to the property.
        3. Explain the key features (BHK options), price, and comprehensive amenities.
        4. State the location and highlight any unique selling points.
        5. Maintain conversational context.
        6. Do NOT list other properties. Focus solely on this one.
        7. CRITICAL: The price_min in the data is in LAKHS. You MUST convert this to CRORES for the user. (e.g., if price_min is 145, write it as '₹1.45 Cr'). Never display values above 100 as 'Lakhs' or 'Crores' without dividing by 100 first.
        """
    else:
        system_prompt = f"""
        Act as a professional Property Advisor for Brigade Group.
        User Query: {query}
        
        PROPERTIES (VALIDATED MATCHES):
        {json.dumps(result, indent=2)}
        
        RULES:
        1. Start with the header exactly: "Top Matching Properties:"
        2. Format as a numbered list (1., 2., 3.): "Project Name — BHK | Price | Highlights"
        3. Keep it professional and extremely concise.
        4. Maintain the context of a conversation.
        5. CRITICAL: The price_min in the data is in LAKHS. You MUST convert this to CRORES for the user. (e.g., if price_min is 145, write it as '₹1.45 Cr'). Never display values above 100 as 'Lakhs' or 'Crores' without dividing by 100 first.
        """
    
    messages = [SystemMessage(content=system_prompt)]
    response = responder_llm.invoke(messages)
    
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
