FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/models
ENV HF_HOME=/models
ENV XDG_CACHE_HOME=/models

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip \
    && pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install -r requirements.txt

COPY app/ .

CMD ["python3", "main.py"]
