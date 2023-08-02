FROM ubuntu:latest

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

RUN pip install --no-cache-dir accelerate==0.19.0 datasets==2.12.0 numpy==1.21.6 ray==2.5.0  \
    raydp==1.6.0b20230527.dev0 torch==1.13.1 transformers==4.26.0 typing==3.7.4.3  \
    tabulate ray[tune] ray[serve] gradio gymnasium dm-tree scikit-image pydantic==1.10.11  \
    tensorboard einops
RUN pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu
RUN pip install intel_extension_for_pytorch==1.13.100+cpu -f https://developer.intel.com/ipex-whl-stable-cpu
