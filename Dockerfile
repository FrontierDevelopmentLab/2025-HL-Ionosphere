# NASA FDL 2025 Ionosphere - Multi-Architecture
# Works on both x86_64 and ARM64 (Apple Silicon)
#
FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /ioncast

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/