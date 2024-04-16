import json
from functools import lru_cache

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from config import emb_model_path, index_path
from prompt_config import (
    rag_suffix,
    convert_to_cpp_temp,
    example_temp,
    example_related_queries,
    generate_search_query_prompt,
    example_scala_code,
)

import re


@lru_cache()
def get_embedding(model_path):
    embedding = HuggingFaceEmbeddings(
        model_name=model_path,
    )
    return embedding


@lru_cache()
def get_vector_store(emb_model_path, index_path):
    embeddings = get_embedding(emb_model_path)
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db


@lru_cache(maxsize=10)
def retrieve_reference(code_query):
    results = ""
    db = get_vector_store(emb_model_path, index_path)

    if isinstance(code_query, str):
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 1})
        matched_documents = retriever.get_relevant_documents(query=code_query)
        for document in matched_documents:
            results += document.page_content + "\n"
    elif isinstance(code_query, tuple):
        for query in code_query:
            retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1, "lambda_mult": 1})
            query = query.lower()
            query = query.replace("velox", " ")
            matched_documents = retriever.get_relevant_documents(query=query)
            for document in matched_documents:
                results += document.page_content + "\n"

    return results


def extract_code(text):
    print(text)
    pattern = r"```c\+\+|```cpp|\```C\+\+|```"
    parts = re.split(pattern, text)
    return parts[1]


def generate_to_cpp_code(llm, code_text):
    convert_to_cpp_prompt = convert_to_cpp_temp.format(code_text)
    res = llm.invoke(convert_to_cpp_prompt)
    cpp_code = extract_code(res.content)
    with open("record.txt", "w") as f:
        f.write(res.content)

    return res.content, cpp_code


def generate_keywords(llm, code_text):
    keywords_prompt = generate_search_query_prompt.substitute(cpp_code=code_text)
    print(keywords_prompt)
    res = llm.invoke(keywords_prompt)
    json_str = res.content
    json_str = json_str.replace("```", "")
    json_str = json_str.replace("json", "")
    keywords = json.loads(json_str)["Queries"]
    return keywords


def generate_velox_udf(llm, cpp_code, rag_queries=example_related_queries, rag_text=""):
    reference = ""
    if bool(rag_text):
        reference = rag_suffix.format(rag_text)

    prompt = example_temp.substitute(reference=reference, queries=rag_queries, cpp_code=cpp_code)
    if bool(rag_text):
        prompt = prompt + rag_suffix.format(rag_text)
    print("------velox udf------")
    print(prompt)
    res = llm.invoke(prompt)

    with open("record.txt", "a") as f:
        print("-" * 70)
        f.write(res.content)

    return res.content


if __name__ == "__main__":
    openai_model = "deepseek-coder:33b-instruct"
    from langchain_community.chat_models.openai import ChatOpenAI

    llm = ChatOpenAI(
        openai_api_base="http://localhost:11434/v1",
        model_name=openai_model,
        openai_api_key="not_needed",
        # max_tokens=2048,
        streaming=True,
    )

    keywords = (
        "Velox string functions",
        "Velox UDF argument handling",
        "Velox default values for UDFs",
    )
    res = retrieve_reference(keywords)
