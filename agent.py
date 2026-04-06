import os
from dotenv import load_dotenv
load_dotenv()
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tools import search_tool

# Split LLMs: 70b to think/plan reliably, 8b to respond rapidly
planner_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
responder_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

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
    2. Filter Inheritance: If the user specified a location (e.g., Chennai) or BHK in a previous turn, and hasn't changed it, retain those parameters in the JSON output.
    
    EXAMPLE BEHAVIOR:
    History:
    USER: 3 BHK in Chennai
    ASSISTANT: 1. Brigade Icon... 2. Brigade Stellaris...
    LATEST QUERY: tell me about the second one
    OUTPUT:
    {{"query": "Brigade Stellaris", "params": {{"bhk": 3, "locality": "Chennai"}}}}
    
    Return ONLY JSON with 'query' and 'params' (bhk, max_price, locality, intents).
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
