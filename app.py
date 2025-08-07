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
            result = chatbot.ask_question(prompt)
            # If chatbot accidentally returns a string, wrap it
            if isinstance(result, str):
                response = {"answer": result, "sources": []}
            else:
                response = result
        except Exception as e:
            response = {
                "answer": f"Error: {e}",
                "sources": []
            }


    # Append sources to answer if available
    answer_text = response["answer"]
    if response.get("sources"):
        sources = "\n".join([f"- {src}" for src in response["sources"]])
        answer_text += f"\n\n**Referenced Document(s):**\n{sources}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add assistant response
    sources = response.get("sources", [])[:2]
    answer_text = response["answer"]

    # Prepare display string with optional sources
    display_text = answer_text
    if sources:
        display_text += "\n\n**Referenced Document(s):** " + ", ".join(sources)

    # Add assistant response to session
    st.session_state.messages.append({
        "role": "assistant", "content": display_text, "timestamp": timestamp
    })

    # Show in chat
    with st.chat_message("assistant"):
        st.markdown(f"{display_text}\n\n*{timestamp}*")

