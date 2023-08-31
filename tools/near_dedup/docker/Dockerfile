FROM ubuntu:22.04

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    git \
    graphviz \
    openjdk-8-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN pip install pyspark
RUN pip install pyrecdp --pre