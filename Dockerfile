FROM rocm/pytorch:latest

# AMD ROCm Mandated Environment for Match 3 (Hostile RAG Retrieval)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HIP_VISIBLE_DEVICES=0 \
    PYTORCH_ROCM_ARCH=gfx942

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code and application
COPY . /app

# Ensure entrypoint is executable
RUN chmod +x /app/app.py

ENTRYPOINT ["python3", "/app/app.py"]
