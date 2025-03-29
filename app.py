import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))

# Streamlit configuration
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("AI Chatbot")
st.subheader("Built with Streamlit, LangChain & GPT-4")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# User input handling
user_input = st.chat_input("Type your message here...")
if user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.predict(input=user_input)
            st.write(response)
    
    # Add AI response to history
    st.session_state.chat_history.append(AIMessage(content=response))

# Sidebar controls
with st.sidebar:
    st.title("Options")  # Corrected title based on the video content

    if st.button("Clear Chat History"):
        # Clear chat history and reinitialize conversation chain
        st.session_state.chat_history = []
        memory = ConversationBufferMemory(return_messages=True)
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        # Updated to use st.rerun() instead of deprecated st.experimental_rerun()
        st.rerun()
    
    # About section in the sidebar
    st.subheader("About")
    st.markdown("""
    **Frameworks Used:**
    - **Streamlit** for the web interface.
    - **LangChain** for conversational memory.
    - **GPT-4** as the language model.
                
    This chatbot uses memory to remember previous messages and generate contextually relevant responses.
    """)
