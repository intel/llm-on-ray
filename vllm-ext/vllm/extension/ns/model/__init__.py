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

from typing import Dict
import time
import os


# stats for model execution performance
class ModelPerf:
    _METRICS = ["non_execution", "model_execution", "compute_logits", "sample"]

    def __init__(self):
        self._perf_stats: Dict[str, int] = {}
        self._perf_stats_prompt: Dict[str, int] = {}
        self._last_call = 0
        self._call_cnt = 0
        self._call_cnt_prompt = 0
        for metric in ModelPerf._METRICS:
            self._perf_stats[metric] = 0
            self._perf_stats_prompt[metric] = 0

        self._ticks = []
        self._perf_steps = int(os.environ.get("NS_MODEL_PERF_STEPS", "50"))

    def reset(self, prompt: bool = False):
        pass

    def non_execution(self, prompt: bool = False):
        pass

    def model_execution(self, prompt: bool = False):
        pass

    def compute_logits(self, prompt: bool = False):
        pass

    def sample(self, prompt: bool = False):
        pass

    def tick(self):
        pass

    def calc_stats(self, prompt: bool = False):
        pass


class RealModelPerf(ModelPerf):
    def reset(self, prompt: bool = False):
        if prompt:
            self._call_cnt_prompt = 0
            for metric in ModelPerf._METRICS:
                self._perf_stats_prompt[metric] = 0
        else:
            self._call_cnt = 0
            for metric in ModelPerf._METRICS:
                self._perf_stats[metric] = 0
        self._ticks.clear()

    def non_execution(self, prompt: bool = False):
        if self._last_call == 0:
            return
        self._ticks.append(self._last_call)
        self._stats(0, prompt)

    def model_execution(self, prompt: bool = False):
        self._stats(1, prompt)

    def compute_logits(self, prompt: bool = False):
        self._stats(2, prompt)

    def sample(self, prompt: bool = False):
        self._stats(3, prompt)

    def tick(self):
        self._ticks.append(time.perf_counter())

    def calc_stats(self, prompt: bool = False):
        self._last_call = time.perf_counter()
        if prompt:
            self._call_cnt_prompt += 1
            perfs = self._perf_stats_prompt
            call_cnt = self._call_cnt_prompt
        else:
            self._call_cnt += 1
            perfs = self._perf_stats
            call_cnt = self._call_cnt
        if call_cnt == self._perf_steps:
            stats = {key: value / call_cnt for key, value in perfs.items()}
            print(f"===execution_model prompt={prompt}: {stats}")
            self.reset(prompt)

    def _stats(self, metric_idx: int, prompt: bool = False):
        last_tick = self._ticks[len(self._ticks) - 1]
        t1 = time.perf_counter()
        self._ticks.append(t1)
        metric = _ModelPerf._METRICS[metric_idx]
        diff = t1 - last_tick
        if prompt:
            self._perf_stats_prompt[metric] += diff
        else:
            self._perf_stats[metric] += diff


_ModelPerf = RealModelPerf() if "NS_MODEL_PERF_STEPS" in os.environ else ModelPerf()
