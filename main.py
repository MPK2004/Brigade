import os
from agent import graph

def test_agent(query: str):
    print(f"\n--- Query: {query} ---")
    
    result = graph.invoke({"query": query})
    
    print("\n[AI Assistant]:")
    print(result.get("response", "No response found."))
    print("-" * 20)

if __name__ == "__main__":
    queries = [
        "What 3 BHK apartments are available in Chennai?",
        "Are there any projects with a swimming pool?",
        "Tell me about luxury projects in Bangalore.",
        "Luxury 3 BHK in Chennai with open spaces",
        "Projects with 4 BHK in Bangalore under 300 Lakhs",
        "4 BHK in Bangalore under 200 Lakhs",
        "4 BHK in Bangalore under 359 Lakhs"
    ]
    
    for q in queries:
        test_agent(q)
