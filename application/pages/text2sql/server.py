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

from fastapi import APIRouter, FastAPI, Request
from langchain_community.chat_models import ChatOpenAI
from starlette.middleware.cors import CORSMiddleware

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from prompt import generate_prompt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SQLGeneratorAPIRouter(APIRouter):
    def __init__(self):
        super().__init__()
        self.openai_model = "sqlcoder-7b-2"

        self.llm = ChatOpenAI(
            openai_api_base="http://localhost:8000/v1",
            model_name=self.openai_model,
            openai_api_key="not_needed",
            streaming=True,
            max_tokens=512,
        )
        self.embeddings = self.setup_emb_model()
        self.db_retriever = self.setup_db_retriever(self.embeddings)

    def setup_emb_model(self, emb_model_name="defog/sqlcoder-7b-2"):
        embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)
        tokenizer = embeddings.client.tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        return embeddings

    def setup_db_retriever(
        self,
        embeddings,
        db=os.path.join(os.path.abspath(os.path.dirname(__file__)), "retriever.db"),
        top_k_table=1,
    ):
        db = FAISS.load_local(db, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(
            search_type="mmr", search_kwargs={"k": top_k_table, "lambda_mult": 1}
        )
        return retriever

    def retrieve(self, query):
        matched_tables = []
        matched_documents = self.db_retriever.get_relevant_documents(query=query)
        for document in matched_documents:
            page_content = document.page_content
            matched_tables.append(page_content)
        return matched_tables

    def generate_sql_code(self, query, schema):
        prompt = generate_prompt(query, schema)
        res = self.llm.invoke(prompt)
        return res


router = SQLGeneratorAPIRouter()


@router.post("/v1/retrieve_tables")
async def retrieve_tables(request: Request):
    params = await request.json()
    print(f"[SQLGenerator - chat] POST request: /v1/rag/retrieve_tables, params:{params}")
    query = params["query"]
    matched_tables = router.retrieve(query)
    print(f"[SQLGenerator - chat] matched_tables: {matched_tables}")
    return {"matched_tables": matched_tables}


@router.post("/v1/generate_sql_code")
async def generate_sql_code(request: Request):
    params = await request.json()
    print(f"[SQLGenerator - chat] POST request: /v1/rag/generate_sql_code, params:{params}")
    query = params["query"]
    schema = params["schema"]
    sql_code = router.generate_sql_code(query, schema)
    print(f"[SQLGenerator - chat] sql_code: {sql_code}")
    return {"sql_code": sql_code}


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    fastapi_port = os.getenv("FASTAPI_PORT", "8080")
    uvicorn.run(app, host="0.0.0.0", port=int(fastapi_port))
