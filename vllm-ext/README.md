# Inference-Engine

To further speed up quantized model inference on Intel CPU, we extend vLLM to run the model decoding in own own inference engine, which is based on [https://github.com/intel/neural-speed](neural-speed).
Neural Speed is an innovative library designed to support the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) low-bit quantization powered by
[https://github.com/intel/neural-compressor](Intel Neural Compressor). The work is inspired by [https://github.com/ggerganov/llama.cpp](llama.cpp) and further optimized for Intel platforms with our
innovations in [https://arxiv.org/abs/2311.00502](NeurIPS' 2023).

You need to first install llm-on-ray with "vllm-cpu" extra.

```bash
pip install .[vllm-cpu] --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

Then, install the vLLM extension and the inference engine.
```bash
cd vllm-ext
pip install .

```

You can check [../docs/vllm.md](vLLM Doc) for more details.
