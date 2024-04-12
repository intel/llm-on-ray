import os
import streamlit as st

#decorator
def enable_chat_history(current_page):
    # to clear chat history after swtching chatbot
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if current_page not in st.session_state:
        st.session_state[current_page] = {}
    if 'messages' not in st.session_state[current_page]:
        st.session_state[current_page]['messages'] = [{"role": "assistant", "content": "Ask me a question "}]
    return st.session_state[current_page]['messages']

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)
    
def display_upload_image(img_file_buffer, author):
    from PIL import Image
    if img_file_buffer is not None:
        img_file_buffer  = Image.open(img_file_buffer)