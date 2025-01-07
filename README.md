# Image Generation API for Children's Stories

This project provides a containerized API for generating story book-style images using various AI models. It's designed to be easily configurable through environment variables, allowing you to switch between different models and adjust parameters.

## Supported Models

1. Stable Diffusion v1.4 (Default)
   - Good for story book style images
   - Balanced performance and quality
   - Lower resource requirements

2. Stable Diffusion XL
   - Higher quality outputs
   - More resource intensive
   - Better understanding of prompts

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
Simply run:
```bash
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

2. Enable GPU mode:
   - Set `USE_GPU=true` in `.env` file
   - Run the application:
```bash
docker compose up --build
```

## API Endpoints

1. Health Check
```bash
GET /
```

2. Generate Image
```bash
POST /generate
```
Optional body parameter:
```json
{
    "custom_prompt": "Your custom prompt here"
}
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

## Best Practices for Story Book Images

1. Use descriptive, detailed prompts
2. Include style-specific keywords like "colorful", "whimsical", "story book style"
3. Experiment with different guidance scales:
   - 7.5 for balanced results
   - 8-10 for more prompt-adherent images

## Performance Considerations

### CPU Mode (`USE_GPU=false`)
- Processing will be slower
- Suitable for testing and development
- Recommended minimum 8GB RAM
- Consider reducing image dimensions for better performance
- Uses PyTorch's CPU optimizations

### GPU Mode (`USE_GPU=true`)
- Significantly faster processing
- Requires NVIDIA GPU with sufficient VRAM
- Better suited for production use
- Can handle larger image dimensions
- Automatically enables CUDA optimizations

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
