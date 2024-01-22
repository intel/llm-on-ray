FROM vault.habana.ai/gaudi-docker/1.12.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest

ENV LANG=en_US.UTF-8

COPY ../../.. /root/llm-on-ray

RUN pip install -e .[hpu]

WORKDIR /root/llm-ray