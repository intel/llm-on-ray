FROM continuumio/anaconda3

WORKDIR /home/user

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    git \
    build-essential \
    vim \
    htop \
    ssh \
    net-tools

RUN conda update --all
RUN conda install mkl mkl-include 

RUN pip install --upgrade pip
RUN pip install astunparse numpy ninja pyyaml setuptools cmake typing_extensions six requests dataclasses datasets

ENV CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# build pytorch
RUN git clone https://github.com/sywangyi/pytorch && cd pytorch && \ 
    git checkout 1.13+FSDP && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0 && \
    python3 setup.py install

# build torch-ccl
COPY frameworks.ai.pytorch.torch-ccl /home/user/frameworks.ai.pytorch.torch-ccl
RUN cd frameworks.ai.pytorch.torch-ccl && python3 setup.py install

# build accelerate
RUN git clone https://github.com/KepingYan/accelerate && \
    cd accelerate && \
    git checkout FSDP_CPU && \
    pip install .

# build transformers
RUN git clone https://github.com/huggingface/transformers && \
    cd transformers && \
    git checkout 8fb4d0e4b46282d96386c229b9fb18bf7c80c25a && \
    pip install .

# install ray-related libs
RUN pip install -U "ray[default]" && pip install --pre raydp && pip install "ray[tune]" tabulate tensorboard

# enable password-less ssh
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -P '' && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    sed -i 's/#   Port 22/Port 12345/' /etc/ssh/ssh_config && \
    sed -i 's/#Port 22/Port 12345/' /etc/ssh/sshd_config

CMD ["sh", "-c", "service ssh start; bash"]