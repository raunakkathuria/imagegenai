# ImageGen

A flexible, containerized API server for image generation that supports multiple AI models with seamless CPU/GPU switching.

## Overview

ImageGen provides a unified interface for generating images using various AI models. Built for flexibility, it supports both CPU and GPU environments, making it suitable for any deployment scenario from development to production.

## Supported Models

### 1. SDXL (Stable Diffusion XL)
- High-quality image generation
- Configuration: `.env.sdxl`
```bash
cp .env.sdxl .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Features:
- Resolution: 1024x1024
- VAE Tiling enabled
- Best for high-quality, detailed images
- Excellent for photorealistic outputs
- Enhanced image quality

### 2. SDXL Turbo
- Fast image generation
- Configuration: `.env.sdxl-turbo`
```bash
cp .env.sdxl-turbo .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Features:
- Resolution: 512x512
- Fast inference (4-8 steps)
- Best for quick iterations
- Good for real-time applications
- Optimized for speed

### 3. PixArt-α LCM
- Fast, high-quality generation
- Configuration: `.env.pixart`
```bash
cp .env.pixart .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Features:
- Resolution: 512x512
- LCM for fast inference
- No guidance needed
- Best for artistic images
- Excellent style consistency

## Key Features

- Unified API interface for all models
- Seamless CPU/GPU switching
- Docker containerization
- Automatic model caching
- Memory optimization
- Resource cleanup
- Environment-based configuration

## API Usage

### Generate Image
```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "your prompt here"}'
```

Response:
```json
{
  "message": "Image generated successfully",
  "filename": "output/image_20240227_123456_abcd1234.png"
}
```

## Memory Management

Each model configuration includes optimized memory settings:
- GPU Memory Usage: 60% (configurable via MAX_MEMORY)
- Attention Slicing enabled
- VAE Tiling where supported
- CPU Offloading when needed
- Memory-efficient attention

### Memory Optimization Features:
1. Dynamic Memory Allocation
   - Automatic GPU memory management
   - Efficient CPU offloading when needed

2. Memory Efficiency Options:
   - Attention slicing for reduced memory footprint
   - VAE tiling for high-resolution images
   - Sequential CPU offloading
   - Memory-efficient attention mechanisms

3. Cleanup Processes:
   - Automatic cache clearing
   - Proper resource deallocation
   - Garbage collection optimization

## Performance Tips

1. Single Worker Mode (Recommended for most cases):
```bash
WORKERS=1 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

2. Multiple Workers (For high-throughput scenarios):
```bash
WORKERS=2 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

3. Optimization Guidelines:
   - Use appropriate resolution for each model
   - Adjust inference steps based on quality needs
   - Monitor GPU memory usage
   - Enable memory optimizations as needed

## Environment Variables

Common settings across all models:
- `USE_GPU`: Enable/disable GPU usage
- `TORCH_DTYPE`: Precision (float32/float16)
- `MAX_MEMORY`: GPU memory limit
- `WORKERS`: Number of server workers
- `HEIGHT/WIDTH`: Output image dimensions
- `NUM_INFERENCE_STEPS`: Generation steps
- `GUIDANCE_SCALE`: CFG scale

Advanced settings:
- `ENABLE_ATTENTION_SLICING`: Memory optimization
- `ENABLE_VAE_TILING`: For high-res images
- `EMPTY_CACHE_BETWEEN_RUNS`: Memory management
- `ENABLE_MODEL_CPU_OFFLOAD`: Resource optimization

## Model-Specific Notes

1. SDXL:
   - Uses VAE Tiling for high-res images
   - Best with higher inference steps (30-50)
   - Excellent for detailed compositions
   - Supports high-resolution outputs
2. SDXL Turbo:
   - Optimized for speed (4-8 steps)
   - Works well with low inference steps
   - Perfect for rapid prototyping
   - Good quality-speed balance
3. PixArt-α LCM:
   - Uses float32 precision for stability
   - No guidance needed (scale=0.0)
   - Fast inference with LCM
   - Excellent for artistic styles

## Requirements

See `requirements.txt` for full dependencies. Key components:
- Python 3.11+
- PyTorch 2.2+
- diffusers 0.24.0
- transformers 4.35.2
- accelerate 0.25.0
- ftfy 6.1.3 (for caption cleaning)

## Docker Support

The application is fully dockerized with:
- Multi-stage builds for optimization
- GPU support via NVIDIA Container Toolkit
- Volume mounting for persistent storage
- Environment-based configuration

### NVIDIA Setup

1. Install NVIDIA Driver:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535  # or latest version
```

2. Install NVIDIA Container Toolkit:
```bash
# Add NVIDIA package repositories
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

3. Verify Installation:
```bash
# Check NVIDIA driver
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Note: For other distributions or detailed instructions, visit:
- [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)

## Error Handling

The API includes robust error handling:
- Memory management errors
- Model loading issues
- Generation failures
- Resource cleanup
- Proper error reporting

## Development

### Local Setup
1. Clone the repository
2. Install dependencies
3. Copy appropriate .env file
4. Run with docker-compose

### Contributing
- Follow PEP 8 guidelines
- Include tests for new features
- Update documentation
- Submit pull requests

## License
This project is open source and available under the MIT License.
