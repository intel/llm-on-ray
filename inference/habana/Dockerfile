FROM vault.habana.ai/gaudi-docker/1.12.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest

ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/root:/usr/lib/habanalabs/

RUN cd ~ && \
    git clone https://github.com/huggingface/optimum-habana.git && \
    cd optimum-habana/ && git reset --hard b6edce65b70e0fadd5d5f51234700bd1144cd0b0 && pip install -e . && cd ../ && \
    cd ./optimum-habana/examples/text-generation/ && \
    pip install -r requirements.txt && \
    cd ~

RUN pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl && \
    pip install ray[tune,serve] transformers==4.32.0 accelerate==0.22.0 pydantic_yaml

WORKDIR /root/llm-ray