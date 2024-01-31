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
import argparse
from typing import Dict, Union

parser = argparse.ArgumentParser(
    description="Example script to query with single request", add_help=True
)
parser.add_argument(
    "--model_endpoint",
    default="http://127.0.0.1:8000",
    type=str,
    help="Deployed model endpoint.",
)
parser.add_argument(
    "--streaming_response",
    default=False,
    action="store_true",
    help="Whether to enable streaming response.",
)
parser.add_argument(
    "--max_new_tokens", default=None, help="The maximum numbers of tokens to generate."
)
parser.add_argument(
    "--temperature",
    default=None,
    help="The value used to modulate the next token probabilities.",
)
parser.add_argument(
    "--top_p",
    default=None,
    help="If set to float < 1, only the smallest set of most probable tokens \
        with probabilities that add up to `Top p` or higher are kept for generation.",
)
parser.add_argument(
    "--top_k",
    default=None,
    help="The number of highest probability vocabulary tokens to keep \
        for top-k-filtering.",
)

args = parser.parse_args()
prompt = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby"
config: Dict[str, Union[int, float]] = {}
if args.max_new_tokens:
    config["max_new_tokens"] = int(args.max_new_tokens)
if args.temperature:
    config["temperature"] = float(args.temperature)
if args.top_p:
    config["top_p"] = float(args.top_p)
if args.top_k:
    config["top_k"] = float(args.top_k)

sample_input = {"text": prompt, "config": config, "stream": args.streaming_response}

proxies = {"http": None, "https": None}
import time
start_time = time.time()
outputs = requests.post(
    args.model_endpoint,
    proxies=proxies,  # type: ignore
    json=sample_input,
    stream=args.streaming_response,

)
print("---e2e: %s seconds ---" % (time.time() - start_time))

try:
    outputs.raise_for_status()
    if args.streaming_response:
        for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
            print(output, end="", flush=True)
        print()
    else:      
        print(outputs.text, flush=True)
except Exception as e:
    print(e)
