import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from agent import graph

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
        with st.spinner("Finding matches..."):
            # Prepare state
            state = {
                "query": prompt,
                "history": st.session_state.agent_history,
                "tool_args": {},
                "tool_result": [],
                "response": ""
            }
            
            # Execute Graph
            final_state = graph.invoke(state)
            response = final_state["response"]
            
            # Display Response
            st.markdown(response)
            
            # Update Session Histories
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.agent_history.append({"role": "user", "content": prompt})
            st.session_state.agent_history.append({"role": "assistant", "content": response})
