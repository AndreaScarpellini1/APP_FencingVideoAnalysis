# 1. Base image with Python 3.12
FROM python:3.12-slim

# 2. Install system dependencies (ffmpeg useful for video I/O with YOLO)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory inside the container
WORKDIR /app

# 4. Copy requirements first (for better caching)
COPY requirements.txt /app/

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your project into the container
COPY . /app

# 7. Default command to run your app
# Note the capital C and capital M here:
CMD ["python", "Codes/Main.py"]
