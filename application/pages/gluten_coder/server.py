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

from coder import generate_to_cpp_code, generate_keywords, retrieve_reference, generate_velox_udf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GlutenUdfGeneratorAPIRouter(APIRouter):
    def __init__(self):
        super().__init__()
        self.openai_model = "deepseek-coder:33b-instruct"
        self.coder_llm = ChatOpenAI(
            openai_api_base="http://localhost:8000/v1",
            model_name=self.openai_model,
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

    def get_cpp_code(self, code_to_convert):
        answer, cpp_code = generate_to_cpp_code(self.coder_llm, code_to_convert)
        return answer, cpp_code

    def gen_keywords(self, cpp_code):
        velox_keywords = generate_keywords(self.general_llm, cpp_code)
        return velox_keywords

    def retrieve(self, velox_keywords):
        rag_source = retrieve_reference(tuple(velox_keywords))
        return rag_source

    def get_udf(self, code_to_convert, velox_keywords, rag_source):
        udf_answer = generate_velox_udf(
            self.coder_llm,
            code_to_convert,
            rag_queries=",".join(velox_keywords),
            rag_text=rag_source,
        )
        return udf_answer


router = GlutenUdfGeneratorAPIRouter()


@router.post("/v1/convert_to_cpp")
async def convert_to_cpp(request: Request):
    params = await request.json()
    print(f"[GlutenUDFConverter - chat] POST request: /v1/rag/convert_to_cpp, params:{params}")
    code = params["code"]
    answer, cpp_code = router.get_cpp_code(code)
    print(f"[GlutenUDFConverter - chat] answer: {answer}, cpp_code: {cpp_code}")
    return {"answer": answer, "cpp_code": cpp_code}


@router.post("/v1/generate_keywords")
async def keywords(request: Request):
    params = await request.json()
    print(f"[GlutenUDFConverter - chat] POST request: /v1/rag/generate_keywords, params:{params}")
    code = params["cpp_code"]
    velox_keywords = router.gen_keywords(code)
    print(f"[GlutenUDFConverter - chat] velox_keywords: {velox_keywords}")
    return {"velox_keywords": velox_keywords}


@router.post("/v1/retrieve_doc")
async def retrieve_doc(request: Request):
    params = await request.json()
    print(f"[GlutenUDFConverter - chat] POST request: /v1/rag/retrieve_doc, params:{params}")
    velox_keywords = params["velox_keywords"]
    related_docs = router.retrieve(velox_keywords)
    print(f"[GlutenUDFConverter - chat] related_docs: {related_docs}")
    return {"related_docs": related_docs}


@router.post("/v1/get_gluten_udf")
async def get_gluten_udf(request: Request):
    params = await request.json()
    print(f"[GlutenUDFConverter - chat] POST request: /v1/rag/get_gluten_udf, params:{params}")
    velox_keywords = params["velox_keywords"]
    code = params["code"]
    related_docs = params["related_docs"]
    udf_answer = router.get_udf(code, velox_keywords, related_docs)
    print(f"[GlutenUDFConverter - chat] udf_answer: {udf_answer}")
    return {"udf_answer": udf_answer}


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    fastapi_port = os.getenv("FASTAPI_PORT", "8000")
    uvicorn.run(app, host="0.0.0.0", port=int(fastapi_port))
