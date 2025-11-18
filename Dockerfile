# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:24.12-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        tmux \
        pv \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
ARG USERNAME=app
RUN groupadd --gid ${GID} --non-unique ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} --non-unique --create-home --shell /bin/bash ${USERNAME}

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NVIDIA_VISIBLE_DEVICES=all \
    CUDA_VERSION=12.6 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG TORCH_VERSION=2.5.1
ARG CUDA_SUFFIX=cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/${CUDA_SUFFIX}
ARG PYG_WHEEL_URL=https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_SUFFIX}.html

WORKDIR /workspace/trajcast

COPY . ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
        torch==${TORCH_VERSION} \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
    && python -m pip install --no-cache-dir \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        torch-spline-conv \
        pyg-lib \
        -f ${PYG_WHEEL_URL} \
    && python -m pip install --no-cache-dir -e ".[cueq,examples,mdanalysis]"

USER app

CMD ["/bin/bash"]
