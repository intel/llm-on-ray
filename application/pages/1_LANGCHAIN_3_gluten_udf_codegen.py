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
import streamlit as st

from codegen.coder import (
    generate_to_cpp_code,
    generate_velox_udf,
    retrieve_reference,
    generate_keywords,
)

from langchain_community.chat_models import ChatOpenAI

st.set_page_config(page_title="Gluten_Coder_Chatbot_V2", page_icon="💬")
st.header("Gluten Coder Chatbot")
st.write("Convert code to Gluten/Velox UDF with the LLM")
from code_editor import code_editor

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
        self.openai_model = "deepseek-coder:33b-instruct"
        self.coder_llm = ChatOpenAI(
            openai_api_base="http://localhost:8000/v1",
            model_name="deepseek-coder-33b-instruct",
            openai_api_key="not_needed",
            streaming=False,
        )

        self.general_llm = self.coder_llm
        # self.general_llm = ChatOpenAI(
        #     openai_api_base="http://localhost:8000/v1",
        #     model_name="mistral-7b-instruct-v0.2",
        #     openai_api_key="not_needed",
        #     streaming=False,
        # )

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
                cpp_code_res, cpp_code = generate_to_cpp_code(self.coder_llm, code_to_convert)
                with st.chat_message("ai"):
                    st.markdown(cpp_code_res)

            with st.chat_message(name="assistant", avatar="🧑‍💻"):
                st.write(f"Step {step}: Analyze the keywords that may need to be queried")
                step += 1
            with st.spinner("Analyze the  code..."):
                keywords = generate_keywords(self.general_llm, cpp_code)
                with st.chat_message("ai"):
                    st.markdown("\n".join(keywords))

            with st.chat_message(name="assistant", avatar="🧑‍💻"):
                st.write(f"Step {step}: Retrieve related knowledge from velox documentations")
                step += 1
            with st.spinner("Retrieve reference from velox document and code..."):
                rag_source = retrieve_reference(tuple(keywords))
                with st.chat_message("ai"):
                    st.write(rag_source)

            with st.chat_message(name="assistant", avatar="🧑‍💻"):
                st.write(f"Step {step}: Based on the previous analysis, rewrite velox based UDF")
                step += 1
            with st.spinner("Converting the C++ code to velox based udf..."):
                result = generate_velox_udf(
                    self.coder_llm,
                    code_to_convert,
                    rag_queries=",".join(keywords),
                    rag_text=rag_source,
                )
                with st.chat_message("ai"):
                    st.markdown(result)


if __name__ == "__main__":
    obj = Basic()
    obj.main()
