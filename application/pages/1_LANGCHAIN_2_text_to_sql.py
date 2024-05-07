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

import json
import time
import utils
import requests
import streamlit as st
from streaming import StreamHandler

st.set_page_config(page_title="SQL_Chatbot", page_icon="💬")
st.header("SQL Chatbot")
st.write("Allows users to interact with the LLM")


class Basic:
    def __init__(self):
        self.server_url = "http://127.0.0.1:8080"
        self.history_messages = utils.enable_chat_history("basic_chat")

    def _post_parse_response(self, response):
        if response.status_code == 200:
            text = response.text
            json_data = json.loads(text)
            return json_data
        else:
            print("Error Code: ", response.status_code)
            return None

    def main(self):
        for message in self.history_messages:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_query := st.chat_input(placeholder="Ask me anything!"):
            self.history_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st_cb = StreamHandler(st.empty())
                    response_rag = requests.post(
                        f"{self.server_url}/v1/retrieve_tables", json={"query": user_query}
                    )
                    json_data_rag = self._post_parse_response(response_rag)
                    matched_tables = json_data_rag["matched_tables"]

                    response_sql = requests.post(
                        f"{self.server_url}/v1/generate_sql_code",
                        json={"query": user_query, "schema": matched_tables},
                    )
                    json_data_sql = self._post_parse_response(response_sql)
                    sql_answer = json_data_sql["sql_code"]["content"]
                    self.history_messages.append({"role": "assistant", "content": sql_answer})

                    print(sql_answer)
                    for token in sql_answer.split():
                        time.sleep(1)
                        st_cb.on_llm_new_token(token + " ")


if __name__ == "__main__":
    obj = Basic()
    obj.main()
