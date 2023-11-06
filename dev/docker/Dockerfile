FROM ubuntu:22.04

ENV LANG C.UTF-8

WORKDIR /root/llm-ray

RUN apt-get update -y \
    && apt-get install -y build-essential cmake wget curl git vim htop ssh net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# setup env
SHELL ["/bin/bash", "--login", "-c"]

RUN conda init bash && \
    unset -f conda && \
    export PATH=$CONDA_DIR/bin/:${PATH} && \
    conda config --add channels intel && \
    conda install python==3.9

COPY ./requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt -f https://developer.intel.com/ipex-whl-stable-cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# For serving with DeepSpeed
COPY ./inference/deepspeed/requirements.cpu.txt /tmp
RUN pip install -r /tmp/requirements.cpu.txt
RUN ds_report

# Used to invalidate docker build cache with --build-arg CACHEBUST=$(date +%s)
ARG CACHEBUST=1
COPY ./inference/deepspeed/install-oneapi.sh /tmp
RUN /tmp/install-oneapi.sh
