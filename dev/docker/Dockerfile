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

RUN pip install accelerate==0.21.0 datasets==2.14.6 numpy==1.21.6 ray==2.5.0  \
    raydp==1.6.0b20230527.dev0 transformers==4.31.0 typing==3.7.4.3  \
    tabulate ray\[tune\] ray\[serve\] gradio gymnasium dm-tree scikit-image pydantic==1.10.11  \
    tensorboard einops peft==0.5.0

# For PyTorch 1.13
# COPY ./requirements.intel.txt ./
# RUN pip install -r requirements.intel.txt -f https://developer.intel.com/ipex-whl-stable-cpu \
#     -f https://download.pytorch.org/whl/torch_stable.html

# For PyTorch 2.0
COPY ./requirements.intel.cpu.pt2.txt /tmp
RUN pip install -r /tmp/requirements.intel.cpu.pt2.txt -f https://developer.intel.com/ipex-whl-stable-cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# For DeepSpeed
COPY ./requirements.deepspeed.cpu.txt /tmp
# Requirments should be installed before deepspeed, can't put deepspeed in the same file
RUN pip install -r /tmp/requirements.deepspeed.cpu.txt
# 0.10.2 is required to support bloom
RUN pip install deepspeed==0.10.2 \
&& ds_report

# Used to invalidate docker build cache with --build-arg CACHEBUST=$(date +%s)
ARG CACHEBUST=1
COPY ./dev/docker/install-oneapi.sh /tmp
RUN /tmp/install-oneapi.sh
