import json
from tools import search_tool

res = search_tool("hyderabad", {"locality": "hyderabad"})
print("LEN:", len(res))
if len(res) > 0: print(res[0])
