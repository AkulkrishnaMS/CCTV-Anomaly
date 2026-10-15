# Use an official PyTorch image with CUDA support pre-installed
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies OpenCV needs even in headless mode
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev git && rm -rf /var/lib/apt/lists/*RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching --
# only re-installs if requirements.txt actually changes)
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy your actual project code
COPY app/ ./app/
COPY config/ ./config/
COPY run_pipeline.py .

# MNAD repo -- clone fresh at build time so it's always the exact
# version you're using, rather than copying a potentially stale local copy
RUN git clone https://github.com/cvlab-yonsei/MNAD.git ./MNAD

# These will be mounted as volumes at runtime, not baked into the image
RUN mkdir -p /app/datasets /app/checkpoints /app/outputs /app/logs

ENTRYPOINT ["python", "run_pipeline.py"]
