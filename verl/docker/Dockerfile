FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.11     (conda)
# pytorch       2.5.1    (pip)
# ==================================================================

ENV PATH=/opt/conda/bin:$PATH
ENV LC_ALL=C
ENV TZ=Asia/Shanghai

COPY miniconda.sh /miniconda.sh
COPY torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl /torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl
# COPY nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl /nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl

# RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    # sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    # sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
RUN chmod 777 /tmp && \
    mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >/etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >>/etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >>/etc/apt/sources.list && \
    echo "deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >>/etc/apt/sources.list && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --default-timeout=500" && \
    GIT_CLONE="git clone --depth 10" && \
    CONDA_INSTALL="conda install -y" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get -y upgrade && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    # DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    $APT_INSTALL \
        apt-utils \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        openssh-client \
        openssh-server \
        libboost-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libjpeg-dev \
        libturbojpeg \
        ffmpeg \
        ninja-build \
        zip \
        curl \
        tmux \
        iputils-ping \
        net-tools \
        openmpi-bin \
        openmpi-common \
        libopenmpi-dev \
        && \
# ==================================================================
# Miniconda3
# ------------------------------------------------------------------
    # wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash /miniconda.sh -b -p /opt/conda && \
        rm /miniconda.sh && \
        #echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
# ==================================================================
# conda
# ------------------------------------------------------------------
    conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes && \
    conda config --set channel_priority flexible && \
    $CONDA_INSTALL \
        python=3.11 && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --upgrade pip && \
    # pip config set global.index-url http://mirrors.aliyun.com/pypi/simple && \
    # pip config set install.trusted-host mirrors.aliyun.com && \
    # $PIP_INSTALL nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl && \
    $PIP_INSTALL torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl && \
    # $PIP_INSTALL flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && \
    # $PIP_INSTALL torch==2.3.0 torchvision==0.18.0 && \
    $PIP_INSTALL \
        packaging \
        ninja \
        lxml \
        jieba \
        av \
        tqdm \
        fire \
        ipdb \
        numpy==1.26.4 \
        wandb \
        isort \
        pyzmq \
        black \
        boto3 \
        scipy \
        awscli \
        psutil \
        dacite \
        joblib \
        loguru \
        openai \
        pyyaml \
        pandas \
        librosa \
        megfile \
        imageio \
        isodate \
        uvicorn \
        fastapi \
        pydantic \
        pendulum \
        requests \
        openpyxl \
        beartype \
        tabulate \
        influxdb \
        colorama \
        jsonlines \
        shortuuid \
        netifaces \
        omegaconf \
        matplotlib \
        smart_open \
        ema_pytorch \
        Levenshtein \
        pycocotools \
        pyinstrument \
        editdistance \
        nvidia-ml-py3 \
        rouge-chinese \
        opencv-python \
        sse-starlette \
        markdown2[all] \
        pytorch-warmup \
        gateloop_transformer \
        webdataset \
        codetiming \
        dill \
        liger-kernel \
        pybind11 \
        pylatexenc \
        ray[default] \
        torchdata \
        mathruler \
        math-verify \
        pytest \
        qwen-vl-utils[decord] \
        aiohttp_cors \
        opencensus \
        tiktoken \
        protobuf \
        modelscope \
        transformers_stream_generator \
        tyro==0.8.14 \
        tensordict==0.5.0 \
        peft==0.15.0 \
        redis==5.0.4 \
        timm==0.6.13 \
        httpx==0.28.1 \
        einops==0.7.0 \
        gradio==5.21.0 \
        gradio_client==1.7.2 \
        Flask==3.0.3 \
        kornia==0.7.0 \
        msgpack==1.0.8 \
        deepspeed==0.16.4 \
        datasets==3.4.1 \
        netifaces==0.11.0 \
        diffusers==0.32.2 \
        hydra-core==1.3.2 \
        tokenizers==0.21.0 \
        urllib3==1.26.18 \
        pycocoevalcap==1.2 \
        einops-exts==0.0.4 \
        bitsandbytes==0.41.0 \
        accelerate==1.5.2 \
        transformers==4.50.0 \
        sentencepiece==0.2.0 \
        huggingface-hub==0.29.3 \
        websocket-client==0.43.0 \
        vllm==0.7.3 \
        trl==0.9.6 \
        && \
    pip --no-cache-dir install --default-timeout=1000 flash-attn --no-build-isolation && \
    pip cache purge && \
    rm /torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl && \
    # rm /nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl && \
    # rm /flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && \
    # rm -r /webdataset && \
    conda clean -y --all && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/* && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    # sed -i 's/#Port 22/Port 8150/' /etc/ssh/sshd_config && \
    # sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    # sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    mkdir -p /root/.ssh && \
    ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -N "" && \ 
    chmod 700 /root/.ssh && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile && \
    echo 'export LC_ALL=C' >> /etc/profile && \
    echo 'export TZ=Asia/Shanghai' >> /etc/profile

COPY llm_engine.py /opt/conda/lib/python3.11/site-packages/vllm/engine/llm_engine.py
COPY scheduler.py /opt/conda/lib/python3.11/site-packages/vllm/core/scheduler.py


# EXPOSE 8150

# ENTRYPOINT ["/bin/bash", "-c", "/usr/sbin/sshd -D & bash"]