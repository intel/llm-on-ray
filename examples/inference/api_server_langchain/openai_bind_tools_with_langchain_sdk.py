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

import argparse
import os

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler, StdOutCallbackHandler

parser = argparse.ArgumentParser(
    description="Example script of enable langchain agent", add_help=True
)
parser.add_argument(
    "--model_name",
    default="mistral-7b-instruct-v0.2",
    type=str,
    help="The name of model to request",
)
parser.add_argument(
    "--streaming_response",
    default=False,
    action="store_true",
    help="Whether to enable streaming response",
)
parser.add_argument(
    "--max_tokens",
    default="512",
    type=int,
    help="max number of tokens used in this example",
)

args = parser.parse_args()

if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.environ["OPENAI_API_KEY"]
else:
    openai_api_key = "not_needed"

if "OPENAI_BASE_URL" in os.environ:
    openai_base_url = os.environ["OPENAI_BASE_URL"]
elif openai_api_key == "not_needed":
    openai_base_url = "http://localhost:8000/v1"
else:
    openai_base_url = "https://api.openai.com/v1"

# =================================================#
# ================================================ #
# Lets define a function/tool for getting the weather. In this demo it we mockthe output
# In real life, you'd end up calling a library/API such as PWOWM (open weather map) library:
# Depending on your app's functionality, you may also, call vendor/external or internal custom APIs

from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool


def get_current_weather(location, unit):
    # Call an external API to get relevant information (like serpapi, etc)
    # Here for the demo we will send a mock response
    weather_info = {
        "location": location,
        "temperature": "78",
        "unit": unit,
        "forecast": ["sunny", "with a chance of meatballs"],
    }
    return weather_info


class GetCurrentWeatherCheckInput(BaseModel):
    # Check the input for Weather
    location: str = Field(
        ..., description="The name of the location name for which we need to find the weather"
    )
    unit: str = Field(..., description="The unit for the temperature value")


class GetCurrentWeatherTool(BaseTool):
    name = "get_current_weather"
    description = "Used to find the weather for a given location in said unit"

    def _run(self, location: str, unit: str):
        # print("I am running!")
        weather_response = get_current_weather(location, unit)
        return weather_response

    def _arun(self, location: str, unit: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = GetCurrentWeatherCheckInput


# ================================================ #

tools = [GetCurrentWeatherTool()]
prompt = ChatPromptTemplate.from_messages([("human", "{input}")])

model = ChatOpenAI(
    openai_api_base=openai_base_url,
    model_name=args.model_name,
    openai_api_key=openai_api_key,
    callbacks=[
        StreamingStdOutCallbackHandler() if args.streaming_response else StdOutCallbackHandler()
    ],
    max_tokens=args.max_tokens,
    streaming=args.streaming_response,
).bind_tools(tools)

runnable = prompt | model

response = runnable.invoke({"input": "what is the weather today in Boston?"})

print(response)
