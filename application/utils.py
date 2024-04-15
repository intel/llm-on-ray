#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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