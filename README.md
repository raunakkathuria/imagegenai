# ImageGen

A flexible, containerized API server for image generation that supports multiple AI models with seamless CPU/GPU switching.

## Overview

ImageGen provides a unified interface for generating images using various AI models. Built for flexibility, it supports both CPU and GPU environments, making it suitable for any deployment scenario from development to production.

## Supported Models

1. Stable Diffusion v1.4 (Default)
   - Versatile general-purpose model
   - Balanced performance and quality
   - Lower resource requirements

2. Stable Diffusion XL
   - Enhanced image quality
   - Advanced prompt understanding
   - Higher resource requirements

The system is designed to be extensible, allowing for easy integration of additional models.

## Requirements

### Minimum Requirements
- Docker
- 8GB RAM (16GB recommended)

### Optional Requirements (for GPU acceleration)
- NVIDIA GPU
- NVIDIA Container Toolkit

## Setup Guide

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Configure the environment variables in `.env` file:
```bash
# Runtime Configuration
USE_GPU=false  # Set to true for GPU acceleration

# Model Selection
MODEL_TYPE=stable-diffusion
MODEL_ID=CompVis/stable-diffusion-v1-4

# Model Parameters
GUIDANCE_SCALE=7.5
NUM_INFERENCE_STEPS=50
HEIGHT=512
WIDTH=512

# GPU Memory Optimization
ENABLE_ATTENTION_SLICING=true
ENABLE_MEMORY_EFFICIENT_ATTENTION=true
```

3. Running the Application:

### CPU Mode (Default)
```bash
# Set USE_GPU=false in .env file
docker compose up --build
```

### GPU Mode
1. First-time GPU Setup (Ubuntu):
```bash
# Install NVIDIA Driver (if not installed)
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
# Reboot required after driver installation

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU setup
nvidia-smi
```

2. Run with GPU support:
   - Set `USE_GPU=true` in `.env` file
   - Use the GPU compose override:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The application uses Docker Compose override files to manage GPU configuration:
- `docker-compose.yml`: Base configuration for CPU mode
- `docker-compose.gpu.yml`: Additional configuration for GPU support

## API Usage

### API Endpoints

1. Health Check
```bash
# Check if the API is running
curl http://localhost:8080/
```
Expected response:
```json
{
    "message": "Image Generation API is running"
}
```

2. Generate Image
```bash
# Generate image with default prompt
curl -X POST http://localhost:8080/generate

# Generate image with custom prompt
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "A dramatic coastal sunset with crashing waves against cliffs, vibrant orange and purple sky, photorealistic style, high detail, cinematic lighting"}'
```
Expected response:
```json
{
    "message": "Image generated successfully",
    "filename": "output/image_20240227_123456_abcd1234.png"
}
```

The generated image will be saved in the `output` directory with a timestamp and unique identifier.

### Example Prompts

Here are some example prompts demonstrating different styles:

1. Photorealistic:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "A modern city skyline at sunset with dramatic clouds, photorealistic, detailed, golden hour lighting, 4k, high definition"}'
```

2. Digital Art:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "A futuristic space station orbiting a purple nebula, digital art, vibrant colors, detailed, sci-fi concept art"}'
```

3. Abstract:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "Abstract geometric patterns flowing like liquid, neon colors, minimalist design, modern art style"}'
```

## Model Configuration Guide

### Stable Diffusion v1.4
```env
MODEL_TYPE=stable-diffusion
MODEL_ID=CompVis/stable-diffusion-v1-4
```

### Stable Diffusion XL
```env
MODEL_TYPE=stable-diffusion-xl
MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
```

## Parameters Guide

- `GUIDANCE_SCALE`: Controls how closely the image follows the prompt (default: 7.5)
  - Higher values (8-15): More closely follows prompt but may be less creative
  - Lower values (5-7): More creative but may stray from prompt

- `NUM_INFERENCE_STEPS`: Number of denoising steps (default: 50)
  - Higher values: Better quality but slower generation
  - Lower values: Faster generation but lower quality

- `HEIGHT` and `WIDTH`: Image dimensions (default: 512x512)
  - Must be multiples of 8
  - Larger sizes require more VRAM

## Output

Generated images are saved in the `output` directory with timestamps and unique identifiers.

## Testing Different Models

1. Update the `.env` file with desired model and parameters
2. Restart the container:
```bash
docker compose down
docker compose up --build
```

## Best Practices for Image Generation

1. Use clear, detailed prompts
2. Include style keywords for desired aesthetics
3. Experiment with different parameters:
   - Guidance Scale: 7.5 for balanced results, 8-10 for strict prompt adherence
   - Steps: Higher for quality, lower for speed
   - Resolution: Adjust based on your needs and resources

## Performance Considerations

### Runtime Modes

#### CPU Mode (`USE_GPU=false`)
- Default mode for maximum compatibility
- Processing will be slower but reliable
- Suitable for:
  * Development and testing
  * Systems without GPU
  * Environments where GPU setup is complex
- Recommended minimum 8GB RAM
- Consider reducing image dimensions for better performance
- Uses PyTorch's CPU optimizations

#### GPU Mode (`USE_GPU=true`)
- Significantly faster processing when available
- Requires NVIDIA GPU with sufficient VRAM
- Better suited for production use
- Can handle larger image dimensions
- Features:
  * Automatic CPU fallback if GPU is unavailable
  * Seamless recovery from GPU errors
  * Dynamic optimization based on available resources
- Fallback scenarios (automatic CPU switch):
  * NVIDIA drivers not found
  * GPU memory insufficient
  * CUDA initialization fails
  * Runtime GPU errors

The system will automatically handle transitions between GPU and CPU:
1. If GPU is requested but unavailable, falls back to CPU
2. If GPU fails during operation, automatically retries on CPU
3. Continues to work on CPU even after GPU failures

### Safety and Optimization Features
- SafeTensors model loading for improved security
- Content safety filters enabled by default
- Automatic memory optimization:
  * Attention slicing for memory efficiency
  * VAE slicing for SDXL models
  * CPU offloading when needed
  * FP16 precision on GPU for better performance

## Troubleshooting

1. GPU Issues:
   - Verify NVIDIA driver installation: `nvidia-smi`
   - Check NVIDIA Container Toolkit: `nvidia-ctk --version`
   - If `USE_GPU=true` but GPU is unavailable, the system will automatically fall back to CPU mode
   - Check logs for device information and optimization status
   - For "Error: unknown runtime specified nvidia", restart Docker: `sudo systemctl restart docker`

2. Memory Issues:
   - Reduce image dimensions
   - Use memory efficient attention for SDXL
   - Reduce batch size if implementing batch processing

3. Security and Dependencies:
   - The project uses the latest stable versions of all dependencies for security
   - Key security features:
     * SafeTensors model loading for improved security
     * Latest ML libraries with security patches
     * Content safety filters enabled
     * Regular dependency updates recommended
   - If you encounter any issues:
     ```bash
     # Rebuild the container to get latest updates
     docker compose down
     docker compose up --build
     ```
   - For manual environment setup:
     ```bash
     pip install -r requirements.txt
     ```
   - Performance optimizations:
     * Automatic FP16 precision for GPU
     * Memory-efficient attention mechanisms
     * CPU offloading for better resource management
     * VAE slicing for SDXL models
   - Safety considerations:
     * Models include content filtering
     * Logs provide visibility into generation process
     * Error handling with graceful fallbacks

4. Best Practices:
   - Regularly update dependencies for security patches
   - Monitor GPU memory usage with nvidia-smi
   - Use version control for your .env configurations
   - Keep Docker and NVIDIA drivers up to date
