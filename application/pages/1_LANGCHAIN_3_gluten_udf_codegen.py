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
import requests
import streamlit as st
from code_editor import code_editor
import json

st.set_page_config(page_title="Gluten_Coder_Chatbot_V2", page_icon="💬")
st.header("Gluten Coder Chatbot")
st.write("Convert code to Gluten/Velox UDF with the LLM")

code_editor_btns_config = [
    {
        "name": "Copy",
        "feather": "Copy",
        "hasText": True,
        "alwaysOn": True,
        "commands": [
            "copyAll",
            [
                "infoMessage",
                {"text": "Copied to clipboard!", "timeout": 2500, "classToggle": "show"},
            ],
        ],
        "style": {"top": "0rem", "right": "0.4rem"},
    },
    {
        "name": "Run",
        "feather": "Play",
        "primary": True,
        "hasText": True,
        "showWithIcon": True,
        "commands": ["submit"],
        "style": {"bottom": "0.44rem", "right": "0.4rem"},
    },
]

info_bar = {
    "name": "input code",
    "css": "\nbackground-color: #bee1e5;\n\nbody > #root .ace-streamlit-dark~& {\n   background-color: #444455;\n}\n\n.ace-streamlit-dark~& span {\n   color: #fff;\n    opacity: 0.6;\n}\n\nspan {\n   color: #000;\n    opacity: 0.5;\n}\n\n.code_editor-info.message {\n    width: inherit;\n    margin-right: 75px;\n    order: 2;\n    text-align: center;\n    opacity: 0;\n    transition: opacity 0.7s ease-out;\n}\n\n.code_editor-info.message.show {\n    opacity: 0.6;\n}\n\n.ace-streamlit-dark~& .code_editor-info.message.show {\n    opacity: 0.5;\n}\n",
    "style": {
        "order": "1",
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "center",
        "width": "100%",
        "height": "2.5rem",
        "padding": "0rem 0.6rem",
        "padding-bottom": "0.2rem",
        "margin-bottom": "-1px",
        "borderRadius": "8px 8px 0px 0px",
        "zIndex": "9993",
    },
    "info": [{"name": "Your code", "style": {"width": "800px"}}],
}


class Basic:
    def __init__(self):
        self.server_url = "http://127.0.0.1:8000"

    def _post_parse_response(self, response):
        if response.status_code == 200:
            text = response.text
            json_data = json.loads(text)
            return json_data
        else:
            print("Error Code: ", response.status_code)
            return None

    def main(self):
        step = 1

        response_dict = code_editor(
            "",
            height=(8, 20),
            lang="scala",
            theme="dark",
            shortcuts="vscode",
            focus=False,
            buttons=code_editor_btns_config,
            info=info_bar,
            props={"style": {"borderRadius": "0px 0px 8px 8px"}},
            options={"wrap": True},
        )
        code_to_convert = response_dict["text"]

        if bool(code_to_convert):
            print(code_to_convert)

            with st.chat_message(name="assistant", avatar="🧑‍💻"):
                st.write(f"Step {step}:  convert the code into C++")
                step += 1
            with st.spinner("Converting your code to C++..."):
                data = {"code": code_to_convert}
                response = requests.post(self.server_url + "/v1/convert_to_cpp", json=data)
                json_data = self._post_parse_response(response)
                cpp_code_res = json_data["answer"]
                cpp_code = json_data["cpp_code"]
                with st.chat_message("ai"):
                    st.markdown(cpp_code_res)

            with st.chat_message(name="assistant", avatar="🧑‍💻"):
                st.write(f"Step {step}: Analyze the keywords that may need to be queried")
                step += 1
            with st.spinner("Analyze the  code..."):
                data = {"cpp_code": cpp_code}
                response = requests.post(self.server_url + "/v1/generate_keywords", json=data)
                json_data = self._post_parse_response(response)
                keywords = json_data["velox_keywords"]
                with st.chat_message("ai"):
                    st.markdown("\n".join(keywords))

            with st.chat_message(name="assistant", avatar="🧑‍💻"):
                st.write(f"Step {step}: Retrieve related knowledge from velox documentations")
                step += 1
            with st.spinner("Retrieve reference from velox document and code..."):
                data = {"velox_keywords": keywords}
                response = requests.post(self.server_url + "/v1/retrieve_doc", json=data)
                json_data = self._post_parse_response(response)
                related_docs = json_data["related_docs"]
                with st.chat_message("ai"):
                    st.write(related_docs)

            with st.chat_message(name="assistant", avatar="🧑‍💻"):
                st.write(f"Step {step}: Based on the previous analysis, rewrite velox based UDF")
                step += 1
            with st.spinner("Converting the C++ code to velox based udf..."):
                data = {
                    "velox_keywords": keywords,
                    "code": code_to_convert,
                    "related_docs": related_docs,
                }
                response = requests.post(self.server_url + "/v1/get_gluten_udf", json=data)
                json_data = self._post_parse_response(response)
                udf_answer = json_data["udf_answer"]
                with st.chat_message("ai"):
                    st.markdown(udf_answer)


if __name__ == "__main__":
    obj = Basic()
    obj.main()
