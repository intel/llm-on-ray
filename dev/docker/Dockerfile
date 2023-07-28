FROM ubuntu:18.04

WORKDIR /home
RUN ls ~/

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y build-essential cmake wget curl git vim htop ssh net-tools
RUN cd ~/

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN /bin/bash /home/Miniconda3-latest-Linux-x86_64.sh -b -p /home/miniconda
ENV PATH /home/miniconda/condabin:$PATH
RUN yes | conda create -n llm_ray python=3.8

# setup env
SHELL ["conda", "run", "-n", "llm_ray", "/bin/bash", "-c"]
ARG http_proxy
ARG https_proxy
RUN pip install accelerate==0.19.0 datasets==2.12.0 numpy==1.21.6 ray==2.5.0 raydp==1.6.0b20230527.dev0 torch==1.13.1 transformers==4.26.0 typing==3.7.4.3 tabulate ray[tune] ray[serve] gradio gymnasium dm-tree scikit-image pydantic==1.10.11 tensorboard einops
RUN pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu
RUN pip install intel_extension_for_pytorch==1.13.100+cpu -f https://developer.intel.com/ipex-whl-stable-cpu

RUN conda init bash
# RUN conda activate llm_ray bash
RUN echo "export http_proxy=${http_proxy}" >> ~/.bashrc
RUN echo "export https_proxy=${https_proxy}" >> ~/.bashrc
RUN echo "conda activate llm_ray" >> ~/.bashrc
RUN source ~/.bashrc

