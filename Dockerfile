# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --fix-missing \
    tzdata \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data \
    && mkdir -p /app/labels \
    && mkdir -p /app/models \
    && mkdir -p /app/outputs \
    && mkdir -p /app/logs

# Copy the application code
COPY model.py .
COPY pipeline.py .

# Copy data directories (they will be mounted as volumes but good to have structure)
COPY data/ ./data/
COPY checkpoints/ ./checkpoints/

# Make sure the pipeline script is executable
RUN chmod +x pipeline.py

# Set default command
CMD ["python", "pipeline.py"]