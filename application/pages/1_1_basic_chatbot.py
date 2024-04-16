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

import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')
st.write('Allows users to interact with the LLM')

class Basic:

    def __init__(self):
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('basic_chat')
    
    def setup_chain(self):
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512)
        chain = ConversationChain(llm=llm, verbose=True)
        return chain
    
    def main(self):
        chain = self.setup_chain()
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_query := st.chat_input(placeholder="Ask me anything!"):
            self.history_messages.append({"role": "user", "content": user_query})
            with st.chat_message('user'):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st_cb = StreamHandler(st.empty())
                    response = chain.run(user_query, callbacks=[st_cb])
                    self.history_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = Basic()
    obj.main()