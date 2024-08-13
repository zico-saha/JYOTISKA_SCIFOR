import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as gen_ai
import io

# Loading environment variables
load_dotenv()

# Configuring Streamlit page settings
st.set_page_config(
    page_title="Gemini ChatBOT",
    page_icon="🤖",
    layout="wide",
)

# Getting the API key from env file.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setting up Google Gemini AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initializing chat session in Streamlit
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Sidebar with buttons for Saving/Downloading and Clearing chats
with st.sidebar:
    st.title("Actions")

    st.markdown("<div style='background-color: #eeeeee; padding: 4px; margin-bottom: 32px; border-radius: 5px;'>",
                unsafe_allow_html=True)

    # Button for saving and downloading chats
    if st.button("Save Chat"):
        chat_history = "\n".join(
            [f"{message.role.capitalize()}: {message.parts[0].text}" for message in st.session_state.chat_session.history]
        )
        buffer = io.BytesIO()
        buffer.write(chat_history.encode('utf-8'))
        buffer.seek(0)
        st.download_button(
            label="Download Chat",
            data=buffer,
            file_name="chat_history.txt",
            mime="text/plain",
            key="download"
        )

    # Button for clearing chats
    if st.button("Clear Chat"):
        # Clear chat history functionality
        st.session_state.chat_session.history = []
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Chatbot's title
st.markdown("<h1 style='color: #4CAF50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;'>🤖 Gemini ChatBot</h1>", unsafe_allow_html=True)

# Displaying the chat history
for message in st.session_state.chat_session.history:
    user_type = translate_role_for_streamlit(message.role)
    if user_type == "user":
        bg_color = "#d1e7dd"
        border_color = "#badbcc"
        text_align = "right"
        bubble_side = "margin-left: auto; margin-right: 0;"
    else:
        bg_color = "#f1f1f1"
        border_color = "#cccccc"
        text_align = "left"
        bubble_side = "margin-right: auto; margin-left: 0;"

    st.markdown(f"""
        <div style='background-color: {bg_color}; border: 2px solid {border_color}; padding: 10px; border-radius: 10px; margin-bottom: 10px; color: #000000; text-align: {text_align}; {bubble_side}; max-width: 70%;'>
            {message.parts[0].text}
        </div>
    """, unsafe_allow_html=True)

# Input field for user's message
user_prompt = st.chat_input("Ask Chatbot...")
if user_prompt:
    # Adding user's message to chat and displaying it
    st.markdown(f"""
        <div style='background-color: #d1e7dd; border: 2px solid #badbcc; padding: 10px; border-radius: 10px; margin-bottom: 10px; color: #000000; text-align: right; margin-left: auto; max-width: 70%;'>
            {user_prompt}
        </div>
    """, unsafe_allow_html=True)

    # Sending user's message to Gemini and get the response
    gemini_response = st.session_state.chat_session.send_message(user_prompt)

    # Displaying Gemini's response
    st.markdown(f"""
        <div style='background-color: #f1f1f1; border: 2px solid #cccccc; padding: 10px; border-radius: 10px; margin-bottom: 10px; color: #000000; text-align: left; margin-right: auto; max-width: 70%;'>
            {gemini_response.text}
        </div>
    """, unsafe_allow_html=True)
