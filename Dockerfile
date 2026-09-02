# CUDA 12.1 runtime + cuDNN8, matching the torch==2.5.1+cu121 wheels in requirements.txt
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.9 (matches Python 3.9.23) via deadsnakes, since Ubuntu 22.04 ships 3.10 by
# default. 3.9 reached EOL in Oct 2025, so 3.9.23 is its final release and this stays pinned.
# tini for ssh initial
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
        openssh-server \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && mkdir -p /var/run/sshd \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

EXPOSE 22

WORKDIR /share/automl

COPY requirements.txt .

# torch/torchvision/torchaudio use the +cu121 local version tag, only published on
# PyTorch's own wheel index, so it must be added as an extra index for pip.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# image 只包環境（Python + 套件 + sshd），不烤程式碼跟資料進去——
# 程式碼跟資料一律靠 docker-compose.yml 的 /share bind mount 在執行期即時提供。
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "docker-entrypoint.sh"]

# Pure compute/script environment: keep the container alive so you can
# `docker compose exec automl python your_script.py` or open a shell into it.
# 強行讓 Container 保持開機狀態
CMD ["tail", "-f", "/dev/null"]
