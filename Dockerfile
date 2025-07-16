# NASA FDL 2025 Ionosphere - Multi-Architecture
# Works on both x86_64 and ARM64 (Apple Silicon)
#
FROM --platform=$BUILDPLATFORM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Use LABEL instead of deprecated MAINTAINER
LABEL maintainer="fs <frank.soboczenski@gmail.com>"
LABEL version="2025.1-multiarch"
LABEL description="NASA FDL Astronaut Health - Multi-architecture support"

# Build arguments for cross-compilation
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Set environment variables
ENV PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    screen \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libssl-dev \
    libffi-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install basic scientific packages
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    google-cloud-storage

# Install TensorFlow
RUN pip install --no-cache-dir tensorflow>=2.15.0

# Install PyTorch with architecture-specific CUDA support
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip install --no-cache-dir torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$ARCH" = "aarch64" ]; then \
        pip install --no-cache-dir torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0; \
    else \
        echo "Installing CPU-only PyTorch for unsupported architecture: $ARCH" && \
        pip install --no-cache-dir torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Verify installations
RUN python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" && \
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash worker && \
    mkdir -p /app && \
    chown -R worker:worker /app

WORKDIR /app

# Copy requirements file before switching to non-root user
COPY requirements.txt /app/requirements.txt

# Install packages from requirements.txt as root user (before USER worker)
RUN pip install --no-cache-dir -r requirements.txt

USER worker

CMD ["python", "-c", "print('Multi-arch container ready!'); import time; time.sleep(3600)"]
CMD tail -f /dev/null
