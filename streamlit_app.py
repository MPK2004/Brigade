import streamlit as st
import os
import time
import json
from dotenv import load_dotenv
load_dotenv(override=True)
from agent import graph
import tools
# Shared Resource Caching (Ensures resources load ONLY ONCE and stay in memory)
@st.cache_resource
def get_shared_resources():
    return tools.get_client(), tools.get_model()

client, model = get_shared_resources()

# Page Config
st.set_page_config(page_title="Brigade Property Advisor", page_icon="🏢", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        border-top: 1px solid #30363d;
    }
    .sidebar .sidebar-content {
        background-color: #161b22;
    }
    h1 {
        color: #d4af37; /* Gold Brigade color */
    }
    .property-card {
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.brigadegroup.com/themes/brigade/images/logo.png", width=200)
    st.title("Advisor Controls")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Suggested Queries")
    st.info("• 3 BHK in Chennai")
    st.info("• Luxury projects in Bangalore")
    st.info("• Apartments with open spaces")

# Main Chat UI
st.title("🏢 Brigade Property Advisor")
st.caption("AI-Powered Luxury Real Estate Search")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help you find your home today?"):
    # Add User Message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with Agent
    with st.chat_message("assistant"):
        # 1. UI Status Management
        status = st.status("Thinking...", expanded=True)
        
        # 2. Generator with Dynamic UI Updates
        def response_generator(state, status_box):
            # Stream state updates from LangGraph
            for update in graph.stream(state, stream_mode="updates"):
                for node, values in update.items():
                    if node == "planner":
                        status_box.update(label="🤔 Analyzing your requirements...", state="running")
                    
                    elif node == "tool":
                        status_box.update(label="📂 Accessing Database Source...", state="running")
                        result = values.get("tool_result", [])
                        if result:
                            with status_box:
                                if result == ["CHITCHAT"]:
                                    st.markdown("✨ *Casual query detected. No DB retrieval needed.*")
                                else:
                                    st.markdown(f"#### RAW DATABASE KNOWLEDGE ({len(result)} records)")
                                    for p in result:
                                        st.markdown(f"**Source Record: {p.get('name', 'N/A')}**")
                                        st.code(json.dumps(p, indent=2), language="json")
                    
                    elif node == "responder":
                        response_content = values.get("response", "")
                        if response_content:
                            status_box.update(label="✅ Matches found!", state="complete", expanded=False)
                            # To simulate streaming effect for better UX
                            words = response_content.split(" ")
                            for i in range(len(words)):
                                yield words[i] + " "
                                time.sleep(0.01)

        # Prepare state
        state = {
            "query": prompt,
            "history": st.session_state.agent_history,
            "tool_args": {},
            "tool_result": [],
            "response": "",
            "client": client,
            "model": model
        }
        
        # 4. Stream and collect final response string
        full_response = st.write_stream(response_generator(state, status))
        
        # Update Session Histories
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.agent_history.append({"role": "user", "content": prompt})
        st.session_state.agent_history.append({"role": "assistant", "content": full_response})
