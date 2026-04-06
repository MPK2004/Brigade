import json
from agent import graph

history = [
    {"role": "user", "content": "what about in bangalore?"},
    {"role": "assistant", "content": "1. Brigade Insignia... 2. Brigade Eternia... 3. Brigade Lumina..."},
]

out = graph.invoke({"query": "below 1 crore?", "history": history, "tool_args": {}})
print(out["response"])
