# NASA FDL 2025 Ionosphere - Multi-Architecture
# Works on both x86_64 and ARM64 (Apple Silicon)
#
FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ioncast

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install imageio[ffmpeg] --no-cache-dir

# Ephemeris data used by the skyfield dependency of the sunmoongeometry dataset
RUN wget --show-progress https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp -O de421.bsp

COPY scripts/*.py ./