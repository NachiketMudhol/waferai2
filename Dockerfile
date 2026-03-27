FROM python:3.11-slim

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + model weights
COPY . .

# HuggingFace Spaces always routes external traffic to port 7860
EXPOSE 7860

CMD ["python", "app.py"]
