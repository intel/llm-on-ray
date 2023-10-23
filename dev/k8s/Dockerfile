FROM rayproject/ray:2.5.0

ARG http_proxy
ARG https_proxy

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y build-essential cmake wget curl git vim htop ssh net-tools
RUN $HOME/anaconda3/bin/pip install accelerate==0.19.0
RUN $HOME/anaconda3/bin/pip install datasets==2.12.0
RUN $HOME/anaconda3/bin/pip install numpy==1.21.6
RUN $HOME/anaconda3/bin/pip install ray==2.5.0
RUN $HOME/anaconda3/bin/pip install raydp==1.6.0b20230527.dev0
RUN $HOME/anaconda3/bin/pip install install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN $HOME/anaconda3/bin/pip install transformers==4.26.0
RUN $HOME/anaconda3/bin/pip install typing==3.7.4.3
RUN $HOME/anaconda3/bin/pip install tabulate
RUN $HOME/anaconda3/bin/pip install ray[tune]
RUN $HOME/anaconda3/bin/pip install ray[serve]
RUN $HOME/anaconda3/bin/pip install gradio
RUN $HOME/anaconda3/bin/pip install gymnasium
RUN $HOME/anaconda3/bin/pip install dm-tree
RUN $HOME/anaconda3/bin/pip install scikit-image
RUN $HOME/anaconda3/bin/pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu

# set http_proxy & https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
