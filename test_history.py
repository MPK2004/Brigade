import os
from dotenv import load_dotenv
load_dotenv()
from agent import graph

def test_memory():
    # Helper to simulate a multi-turn conversation
    history = []
    
    queries = [
        "What are the luxury properties in Bangalore?",
        "tell me more about the 2nd one",
        "any with a pool?"
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
        
        # We need to manually maintain the "messages" history format used by the planner
        # We'll just append what we think happened
        
        final_state = graph.invoke(state)
        response = final_state["response"]
        
        print(f"[AI Assistant]:\n{response}")
        
        # update history
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    test_memory()
