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
RUN pip install astunparse nltk numpy ninja pyyaml setuptools cmake typing_extensions six requests dataclasses datasets evaluate scikit-image dm-tree gymnasium

ENV CMAKE_PREFIX_PATH=/opt/conda

# build pytorch
RUN git clone https://github.com/sywangyi/pytorch && cd pytorch && \ 
    git checkout 1.13+FSDP && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0 && \
    python setup.py install

# build torch-ccl
RUN git clone https://github.com/intel/torch-ccl.git && cd torch-ccl && \ 
    git checkout ccl_torch1.13.100 && \ 
    git submodule sync && \ 
    git submodule update --init --recursive && \ 
    python setup.py install

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
RUN pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"
RUN pip install --pre raydp && pip install "ray[tune]" tabulate tensorboard

# install java
RUN wget --no-check-certificate -q https://repo.huaweicloud.com/java/jdk/8u201-b09/jdk-8u201-linux-x64.tar.gz && \
    tar -zxvf jdk-8u201-linux-x64.tar.gz && \
    mv jdk1.8.0_201 /opt/jdk1.8.0_201 && \
    rm jdk-8u201-linux-x64.tar.gz

# install hadoop 3.3.3
RUN wget --no-check-certificate -q https://dlcdn.apache.org/hadoop/common/hadoop-3.3.3/hadoop-3.3.3.tar.gz && \
    tar -zxvf hadoop-3.3.3.tar.gz && \
    mv hadoop-3.3.3 /opt/hadoop-3.3.3 && \
    rm hadoop-3.3.3.tar.gz

ENV HADOOP_HOME=/opt/hadoop-3.3.3
ENV JAVA_HOME=/opt/jdk1.8.0_201
ENV JRE_HOME=$JAVA_HOME/jre
ENV PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
ENV HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
ENV HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native"
ENV LD_LIBRARY_PATH=$HADOOP_HOME/lib/native

# enable password-less ssh
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -P '' && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    sed -i 's/#   Port 22/Port 12345/' /etc/ssh/ssh_config && \
    sed -i 's/#Port 22/Port 12345/' /etc/ssh/sshd_config

CMD ["sh", "-c", "service ssh start; bash"]

