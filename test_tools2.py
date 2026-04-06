from tools import search_tool
res = search_tool("lumina")
print(res[0]['name'], "price_min:", res[0]['price_min'])
