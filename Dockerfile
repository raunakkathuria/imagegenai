FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create only output directory (since others will be volume mounted)
RUN mkdir -p output && chmod 777 output

# Copy application code
COPY . .

EXPOSE 8080
CMD ["python3", "app.py"]
