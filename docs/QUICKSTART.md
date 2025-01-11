# ImageGenAI Quickstart Guide

This guide will help you get started with the ImageGenAI API, a flexible service that supports multiple AI image generation models.

## Prerequisites

1. System Requirements:
   - Docker and Docker Compose
   - For GPU models:
     * NVIDIA GPU with CUDA support
     * NVIDIA Container Toolkit

2. NVIDIA Setup (Ubuntu/Debian) - Required only for GPU models:
```bash
# Install NVIDIA Driver
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd ImageGenAI
```

2. Setup directories and clean cache:
```bash
# Remove old cache if exists (important when changing configurations)
rm -rf cache models

# Create fresh directories with proper permissions
mkdir -p cache models output
chmod 777 cache models output  # Ensure Docker has write permissions
```

3. Choose a model configuration:
```bash
# For CPU-only mode (no GPU required)
cp .env.cpu .env
docker compose -f docker-compose.yml up --build

# For GPU models:
# SDXL base model
cp .env.sdxl .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# SDXL Turbo (faster)
cp .env.sdxl-turbo .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# PixArt-α LCM
cp .env.pixart .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

4. Generate images:

Basic usage:
```bash
# Using environment defaults
curl -X POST "http://localhost:8080/generate"

# With custom prompt
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "custom_prompt": "your prompt here"
  }'
```

Advanced usage with runtime parameters:
```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "custom_prompt": "your prompt here",
    "negative_prompt": "things to avoid",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "height": 768,
    "width": 768
  }'
```

Parameter ranges:
- num_inference_steps: 1-150 steps
- guidance_scale: 0.0-20.0
- height/width: 128-1024 (must be multiple of 8)

## Supported Models

### 1. CPU Mode (Stable Diffusion v1.5)
- Best for: Systems without GPU
- Resolution: 512x512
- Configuration: `.env.cpu`
- Use case: Development and testing
- Model: runwayml/stable-diffusion-v1-5

### 2. SDXL (Stable Diffusion XL)
- Best for: High-quality, detailed images
- Resolution: 1024x1024
- Configuration: `.env.sdxl`
- Use case: Professional quality images

### 3. SDXL Turbo
- Best for: Quick generations
- Resolution: 512x512
- Configuration: `.env.sdxl-turbo`
- Use case: Rapid prototyping, real-time applications

### 4. PixArt-α LCM
- Best for: Artistic images
- Resolution: 768x768
- Configuration: `.env.pixart`
- Use case: Creative and artistic outputs

## Configuration Guide

### Common Settings

```env
# Device Settings
USE_GPU=true  # Set false for CPU mode
TORCH_DTYPE=float32  # or float16 for less memory usage

# Memory Management
MAX_MEMORY=0.6  # Use 60% of GPU memory
ENABLE_ATTENTION_SLICING=true
ENABLE_VAE_TILING=true
EMPTY_CACHE_BETWEEN_RUNS=true

# Generation Parameters (can be overridden via API)
HEIGHT=512  # Image height
WIDTH=512   # Image width
NUM_INFERENCE_STEPS=20  # More steps = more detail
GUIDANCE_SCALE=7.5     # Higher = more prompt adherence
```

### Model-Specific Tips

1. CPU Mode (SD v1.5):
   - Use 15-20 inference steps
   - Enable CPU offloading
   - Good for testing and development

2. SDXL Base:
   - Use higher inference steps (30-50)
   - Enable VAE tiling for high-res
   - Works well with detailed prompts

3. SDXL Turbo:
   - Use 4-8 inference steps
   - Good with float16 precision
   - Best for quick iterations

4. PixArt-α LCM:
   - Use float32 precision
   - No guidance needed (scale=0.0)
   - Great for artistic styles

## Resource Management

### Disk Space Management

The service uses local directories for caching models and Hugging Face files:
```
./cache  - HuggingFace cache (~22GB for full downloads)
./models - Model cache
./output - Generated images
```

Important: When changing model configurations or updating the code, clean the cache:
```bash
# Stop containers first
docker compose down

# Clean cache directories
rm -rf cache/* models/*

# Rebuild and start
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build  # For GPU models
# or
docker compose -f docker-compose.yml up --build  # For CPU mode
```

Monitor disk usage:
```bash
# Check cache sizes
du -sh ./cache ./models ./output

# Monitor during usage
watch -n 10 'du -sh ./cache ./models ./output'
```

Best practices:
- Clean cache when switching models
- Keep only needed models
- Archive old outputs
- Monitor disk space usage

### Memory Management

#### GPU Memory Tips (for GPU models)

1. Single Worker Mode (Recommended):
```bash
WORKERS=1 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

2. Memory Settings:
   - Start with MAX_MEMORY=0.6
   - Enable attention slicing
   - Use CPU offloading if needed
   - Clear cache between runs

3. Resolution vs Quality:
   - Lower resolution = less memory
   - Fewer steps = faster generation
   - Balance based on your needs

### Common Issues

1. Out of Memory (OOM):
   - Reduce MAX_MEMORY
   - Lower resolution
   - Enable memory optimizations
   - Use float16 precision (except PixArt)
   - Consider CPU mode

2. Slow Generation:
   - Check GPU utilization
   - Reduce inference steps
   - Consider SDXL Turbo
   - Optimize prompt length

3. Model Loading Issues:
   - Clean cache directories
   - Rebuild containers
   - Check error logs
   - Verify configurations

## API Usage

### Generate Image

Basic request:
```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "custom_prompt": "A majestic mountain landscape at sunset, ultra detailed"
  }'
```

Advanced request with parameters:
```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "custom_prompt": "A majestic mountain landscape at sunset",
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "height": 768,
    "width": 768
  }'
```

Response:
```json
{
  "message": "Image generated successfully",
  "filename": "output/image_20240227_123456_abcd1234.png"
}
```

### Health Check
```bash
curl http://localhost:8080/
```

## Best Practices

1. Prompting:
   - Be specific and detailed
   - Use negative prompts for quality
   - Keep prompts concise
   - Test with quick generations first

2. Production Setup:
   - Use proper error handling
   - Monitor memory usage
   - Implement rate limiting
   - Regular cache cleanup

3. Resource Management:
   - Monitor system resources
   - Use appropriate worker count
   - Regular maintenance
   - Proper shutdown handling

## Troubleshooting

1. Model Loading Issues:
   - Clean cache directories
   - Verify CUDA setup (for GPU)
   - Check model cache
   - Review logs carefully

2. Generation Failures:
   - Check memory usage
   - Verify prompt format
   - Check model status
   - Review error messages

3. Performance Issues:
   - Monitor resource usage
   - Check memory settings
   - Verify network speed
   - Review configuration

## Getting Help

- Check the logs: `docker compose logs`
- Review error messages
- Check GPU status: `nvidia-smi` (for GPU mode)
- Monitor memory usage
