from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Model configuration (same as your original code)
model = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Welcome! Chat with Llama 3.2")

# Display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Same exit functionality: type 0 to exit
    if prompt.strip() == "0":
        st.session_state.messages = []
        st.rerun()

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response (same functionality)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(prompt)
            st.markdown(response.content)

    # Store bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })