import os
from dotenv import load_dotenv
load_dotenv(override=True)
from agent import graph

def test_memory():
    history = []
    
    queries = [
        "suggest me property in hyderabad",
        "are these the only ones? also are there any below 1crore",
        "what about in bangalore?",
        "below 1 crore?"
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
        
        print(f"[AI]:\n{response}\n")
        
        history.append({"role": "user", "content": q})
        # Mocking the AI so we don't depend on what it actually replied for the next step, wait, it relies on actual responses?
        # Yes, we pass response to history
        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    test_memory()
