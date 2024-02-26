FROM rayproject/ray:2.5.0

ARG http_proxy
ARG https_proxy

RUN apt-get update -y  \
    && apt-get upgrade -y  \
    && apt-get install -y build-essential cmake wget curl git vim htop ssh net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN "$HOME/anaconda3/bin/pip install accelerate==0.19.0" \
    "$HOME/anaconda3/bin/pip install datasets==2.12.0" \
    "$HOME/anaconda3/bin/pip install numpy==1.21.6"  \
    "$HOME/anaconda3/bin/pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"  \
    "$HOME/anaconda3/bin/pip install ray[tune]"  \
    "$HOME/anaconda3/bin/pip install ray[serve]"  \
    "$HOME/anaconda3/bin/pip install gradio"  \
    "$HOME/anaconda3/bin/pip install install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu"  \
    "$HOME/anaconda3/bin/pip install transformers==4.26.0"  \
    "$HOME/anaconda3/bin/pip install typing==3.7.4.3"  \
    "$HOME/anaconda3/bin/pip install tabulate"  \
    "$HOME/anaconda3/bin/pip install gymnasium"  \
    "$HOME/anaconda3/bin/pip install dm-tree"  \
    "$HOME/anaconda3/bin/pip install scikit-image"  \
    "$HOME/anaconda3/bin/pip install oneccl_bind_pt==1.13 --extra-index-url https://developer.intel.com/ipex-whl-stable-cpu"

# set http_proxy & https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
