import os
from dotenv import load_dotenv
load_dotenv()
from agent import graph

def test_memory():
    history = []
    
    queries = [
        "hi",
        "suggest me a place in whitefield",
    ]
    
    for q in queries:
        print(f"\n--- Turn: {q} ---")
        state = {
            "query": q,
            "history": history,
            "tool_args": {},
            "tool_result": [],
            "response": ""
        }
        
        final_state = graph.invoke(state)
        response = final_state["response"]
        
        print(f"[AI]:\n{response[:200]}...\n")
        
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    test_memory()
