import streamlit as st
from datetime import datetime
from chatbot import CompanyChatbot

# Page config
st.set_page_config(
    page_title="Document AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []

# Load chatbot once
@st.cache_resource
def get_chatbot():
    return CompanyChatbot()

chatbot = get_chatbot()

# Sidebar - Chat History
with st.sidebar:
    st.title("Chat History")

    if st.button("➕ New Chat"):
        # Save current chat if not empty
        if st.session_state.messages:
            st.session_state.saved_chats.append(list(st.session_state.messages))
        # Reset for new chat
        st.session_state.messages = []

    st.markdown("---")
    for i, chat in enumerate(st.session_state.saved_chats):
        if st.button(f"🗂️ Chat {i+1}", key=f"chat_{i}"):
            st.session_state.messages = chat

# Title
st.title("Document AI Assistant")
st.caption("Ask questions about your documents")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"{msg['content']}\n\n*{msg.get('timestamp', '')}*")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add user message
    st.session_state.messages.append({
        "role": "user", "content": prompt, "timestamp": timestamp
    })

    with st.chat_message("user"):
        st.markdown(f"{prompt}\n\n*{timestamp}*")

    with st.spinner("Thinking..."):
        try:
            response = chatbot.ask_question(prompt)
        except Exception as e:
            response = f"Error: {e}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant", "content": response, "timestamp": timestamp
    })

    with st.chat_message("assistant"):
        st.markdown(f"{response}\n\n*{timestamp}*")