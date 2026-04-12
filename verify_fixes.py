import os
from dotenv import load_dotenv
load_dotenv(override=True)
from tools import search_tool, get_client, get_model
import json

def verify():
    print("--- Verification Started ---")
    
    # 1. Verify Price Multiplier Trap (Brigade Icon)
    results = search_tool("Brigade Icon", {"locality": "Chennai"})
    if results:
        p = results[0]
        name = p.get("name")
        price_min = p.get("price_min")
        print(f"Project: {name}")
        print(f"Extracted Price Min: {price_min} Lakhs")
        if price_min and 800 <= price_min <= 900:
            print("✅ SUCCESS: Brigade Icon price correctly extracted (~8.40 Cr = 840 Lakhs).")
        else:
            print(f"❌ FAILURE: Brigade Icon price incorrect ({price_min}). Expected ~840.")
    else:
        print("❌ FAILURE: Brigade Icon not found.")
    
    print("-" * 20)

    # 2. Verify Plots (Brigade Cherry Blossom)
    results = search_tool("Brigade Cherry Blossom")
    if results:
        p = results[0]
        name = p.get("name")
        prop_type = p.get("property_type")
        print(f"Project: {name}")
        print(f"Property Type: {prop_type}")
        if prop_type == "Plot":
            print("✅ SUCCESS: Brigade Cherry Blossom (Plot) correctly ingested.")
        else:
            print(f"❌ FAILURE: Brigade Cherry Blossom type is {prop_type}, expected 'Plot'.")
    else:
        print("❌ FAILURE: Brigade Cherry Blossom not found.")

    print("-" * 20)

    # 3. Verify Nearby Data (Brigade Eternia)
    results = search_tool("Brigade Eternia")
    if results:
        p = results[0]
        nearby = p.get("nearby", {})
        schools = nearby.get("Schools", [])
        print(f"Project: {p.get('name') or 'Unnamed'}")
        print(f"Nearby Schools Count: {len(schools)}")
        if schools:
            print("✅ SUCCESS: Nearby schools data present for Brigade Eternia.")
            for s in schools:
                print(f"  - {s.get('place')} ({s.get('distance')})")
        else:
            print("❌ FAILURE: Nearby schools data missing for Brigade Eternia.")
    else:
        print("❌ FAILURE: Brigade Eternia not found.")

    print("--- Verification Ended ---")

if __name__ == "__main__":
    verify()
