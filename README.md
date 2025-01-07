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
  -d '{"custom_prompt": "A magical treehouse in a giant oak tree, with twinkling lights, rope bridges, and small woodland creatures visiting, watercolor style, whimsical, children book illustration"}'
```
Expected response:
```json
{
    "message": "Image generated successfully",
    "filename": "output/image_20240227_123456_abcd1234.png"
}
```

The generated image will be saved in the `output` directory with a timestamp and unique identifier.

### Testing Different Prompts

Here are some example prompts you can try:

1. Fantasy Scene:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "A young wizard practicing spells in a cozy library filled with floating books, magical creatures, and sparkling potions, digital art, storybook style"}'
```

2. Nature Scene:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "A peaceful garden with butterflies, friendly bees, and talking flowers having a tea party, soft pastel colors, children book illustration style"}'
```

3. Adventure Scene:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"custom_prompt": "A group of animal friends sailing a paper boat down a winding river, passing by curious fish and friendly river creatures, watercolor style, whimsical"}'
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
- Default fallback mode if GPU is unavailable

### GPU Mode (`USE_GPU=true`)
- Significantly faster processing
- Requires NVIDIA GPU with sufficient VRAM
- Better suited for production use
- Can handle larger image dimensions
- Automatically enables CUDA optimizations
- Automatic fallback to CPU mode if:
  * NVIDIA drivers are not found
  * GPU is not available
  * CUDA initialization fails

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
