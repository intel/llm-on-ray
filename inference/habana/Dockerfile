FROM vault.habana.ai/gaudi-docker/1.13.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.0:latest

ENV LANG=en_US.UTF-8

COPY ../../ /root/llm-on-ray

WORKDIR /root/llm-on-ray

RUN pip install . && \
    pip install --upgrade-strategy eager optimum[habana]
