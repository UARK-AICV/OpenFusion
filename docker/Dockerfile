FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific

RUN apt-get update && apt-get install -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    build-essential \
    cmake \
    curl \
    g++ \
    wget \
    bzip2 \
    git \
    vim \
    tmux \
    htop \
    git \
    zip \
    unzip \
    ca-certificates \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libglu1-mesa \
    libxext6 \
    libxtst6 \
    libxrender1 \
    libxi6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libopencv-dev \
    libyaml-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswresample-dev \
    less \
    groff \
    mpich \
    ninja-build

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install torchshow prettytable
ADD openfusion/zoo/xdecoder_seem/requirements.txt /workspace/openfusion/zoo/xdecoder_seem/requirements.txt
RUN pip install --no-cache-dir -r /workspace/openfusion/zoo/xdecoder_seem/requirements.txt
RUN pip install --no-cache-dir --upgrade open3d==0.17.0
RUN pip uninstall tensorflow -y