FROM nvidia/cuda:11.4.1-devel-ubuntu18.04

RUN set -x && \
    echo "Acquire { HTTP { Proxy \"$HTTP_PROXY\"; }; };" | tee /etc/apt/apt.conf

ARG BUILD_DIR=/app

WORKDIR /app

WORKDIR $BUILD_DIR
COPY requirements.txt $BUILD_DIR

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV CUDA_PATH /usr/local/cuda

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget unzip cmake git \
    python3-dev python3-pip python3-setuptools \
    build-essential ca-certificates

RUN ln -sf $(which python3) /usr/bin/python && \
    ln -sf $(which pip3) /usr/bin/pip

ENV	LD_LIBRARY_PATH	/usr/local/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV	LC_CTYPE en_US.UTF-8
ENV	LANG en_US.UTF-8

ENV	NCCL_ROOT_DIR /usr/local/cuda
ENV	TH_BINARY_BUILD 1
ENV	TORCH_CUDA_ARCH_LIST "3.5;5.0+PTX;5.2;6.0;6.1;7.0;7.5"
ENV	TORCH_NVCC_FLAGS "-Xfatbin -compress-all"

RUN pip install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt \

pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

env TRANSFORMERS_CACHE "/app/cache"

WORKDIR /app