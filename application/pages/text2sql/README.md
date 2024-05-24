# Text2SQL AI chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://langchain-chatbot.streamlit.app/)
## Introduction
### Text2SQL
Text2SQL is a natural language processing (NLP) task that aims to convert a natural language question into a SQL query. The goal of Text2SQL is to enable users to interact with databases using natural language, without the need for a specialized knowledge of SQL. The Text2SQL task is a challenging task for NLP systems, as it requires the system to understand the context of the question, the relationships between the words, and the structure of the SQL query.

### The Text2SQL AI chatbot
This chatbot is an implementation of the Text2SQL task using the LLM-on-Ray service for SQL generation, Langchain for RAG. The chatbot is built using the Streamlit library, which allows for easy deployment and sharing of the chatbot. The chatbot uses a pre-trained [defog/sqlcoder-7b-2](https://huggingface.co/defog/sqlcoder-7b-2) model to generate the SQL query, which is then executed on a remote SQL database. The chatbot is designed to be user-friendly and easy to use, with clear instructions and error messages.

### Ability of this chatbot
The objective of this chatbot application is to assist users by seamlessly transforming their user-defined language, originally designed for their database, into SQL code that adheres to the database standards.

The conversion process is streamlined into the following steps:

1. The chatbot identifies and comprehends the logic of the original natural language, then translates it into an initial SQL code draft.
2. Utilizing the customer's database description, the Language Learning Model (LLM) identifies key terms to construct queries. These queries are related to database schema, tables, columns, and data types. The LLM then outputs the query results in JSON format.
3. With the key items identified from the LLM's output, the chatbot retrieve the database documentation stored in vector database(Faiss) to find relevant information.
4. Drawing from the information in the database documentation, the chatbot generates the final SQL code that is tailored to meet the specifications of user's database.

### Configuration
Currently, we are using LLM Model [defog/sqlcoder-7b-2](https://huggingface.co/defog/sqlcoder-7b-2).
Deployment can be done using LLM-on-Ray with the following command:
```
llm_on_ray-serve --config_file llm_on_ray/inference/models/sqlcoder-7b-2.yaml
```

Before launching the Streamlit application, you need to update the application/pages/1_LANGCHAIN_2_text_to_sql.py with the necessary configuration details:

```
# Provide the path to the FAISS index for database documentation.
retriever.db_path = "" #application/pages/text2sql/retriever.db
```