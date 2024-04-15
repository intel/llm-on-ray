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
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="SQL_Chatbot", page_icon="💬")
st.header('SQL Chatbot')
st.write('Allows users to interact with the LLM')

def generate_prompt(question, schema):
    prompt = """### Instructions:
Your task is convert a question into a SQL query, given a MySQL database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float
- Use LIKE instead of ilike
- Only generate the SQL query, no additional text is required
- Generate SQL queries for MySQL database

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
{schema}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
""".format(question=question, schema=schema)

    return prompt

def rag_retrival(retriever, query):
    matched_tables = []
    matched_documents = retriever.get_relevant_documents(query=query)

    for document in matched_documents:
        page_content = document.page_content
        matched_tables.append(page_content)
    return matched_tables

class Basic:

    def __init__(self):
        self.openai_model = "sqlcoder-7b-2"
        self.history_messages = utils.enable_chat_history('basic_chat')
    
    def setup_chain(self):
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512)
        chain = ConversationChain(llm=llm, verbose=True)
        return chain
    
    def setup_db_retriever(self, db=os.path.join(os.path.abspath(os.path.dirname(__file__)),'retriever.db'), emb_model_name="defog/sqlcoder-7b-2", top_k_table=1):
        embeddings = HuggingFaceEmbeddings(model_name = emb_model_name)
        db = FAISS.load_local(db, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': top_k_table, 'lambda_mult': 1})
        return retriever
    
    def main(self):
        chain = self.setup_chain()
        db_retriever = self.setup_db_retriever()
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
                    schema = rag_retrival(db_retriever, user_query)
                    user_query = generate_prompt(user_query, schema)
                    response = chain.run(user_query, callbacks=[st_cb])
                    self.history_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = Basic()
    obj.main()