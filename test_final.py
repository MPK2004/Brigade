import os
from dotenv import load_dotenv
load_dotenv(override=True)
from agent import graph

def test():
    query = "4 BHK in Bangalore under 359 Lakhs"
    print(f"--- Query: {query} ---")
    
    # State reset
    state = {
        "query": query,
        "action": "",
        "tool_args": {},
        "tool_result": [],
        "response": ""
    }
    
    # Run the graph
    for output in graph.stream(state):
        for key, value in output.items():
            print(f"Node: {key}")
            if key == "tool":
                print(f"Results Found: {len(value.get('tool_result', []))}")
            if key == "responder":
                print(f"\n[AI Assistant]:\n{value['response']}")

if __name__ == "__main__":
    test()
