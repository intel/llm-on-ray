# Gluten UDF converter AI chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://langchain-chatbot.streamlit.app/)
## Introduction
### Gluten
[Gluten](https://github.com/apache/incubator-gluten) is a new middle layer to offload Spark SQL queries to native engines. Gluten can benefit from high scalability of Spark SQL framework and high performance of native libraries.

The basic rule of Gluten's design is that we would reuse spark's whole control flow and as many JVM code as possible but offload the compute-intensive data processing part to native code. Here is what Gluten does:

- Transform Spark's whole stage physical plan to Substrait plan and send to native
- Offload performance-critical data processing to native library
- Define clear JNI interfaces for native libraries
- Switch available native backends easily
- Reuse Spark�s distributed control flow
- Manage data sharing between JVM and native
- Extensible to support more native accelerators


### Ability of this chatbot
The objective of this chatbot application is to assist users by seamlessly transforming their user-defined functions (UDFs), originally designed for Vanilla Spark, into C++ code that adheres to the code standards of Gluten and Velox. This is achieved through the utilization of a Language Learning Model (LLM), which automates the conversion process, ensuring compatibility and enhancing performance within these advanced execution frameworks.

The conversion process is streamlined into the following steps:

1. The chatbot identifies and comprehends the logic of the original Spark UDF code, then translates it into an initial C++ code draft.
2. Utilizing the preliminary C++ code, the Language Learning Model (LLM) identifies key terms to construct queries. These queries are related to Velox's existing function implementations and data types. The LLM then outputs the query results in JSON format.
3. With the keywords from the LLM's output, the chatbot retrieve the Velox documentation stored in vector database(Faiss) to find relevant information.
4. Drawing from the information in the Velox documentation, the chatbot generates the final C++ code that is tailored to meet the specifications of Velox UDFs.


### Configuration

Currently, we are using LLM Model [deepseek-coder-33b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct).
Deployment can be done using LLM-on-Ray with the following command:
```
llm_on_ray-serve --config_file llm_on_ray/inference/models/deepseek-coder-33b-instruct.yaml
```

Before launching the Streamlit application, you need to update the config.py file located at application/pages/codegen/config.py with the necessary configuration details:

```
# Specify the directory where the model 'deepseek-coder-33b-instruct' is stored.
model_base_path = ""
# Provide the path to the FAISS index for Velox documentation.
vs_path = ""
```




