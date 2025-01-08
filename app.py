import os
import logging
import gc
import psutil
import torch.cuda
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import torch
from typing import Optional
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    PixArtAlphaPipeline,
)
from PIL import Image
import uuid
from datetime import datetime
from distutils.util import strtobool

# Load environment variables
load_dotenv()

app = FastAPI()

# Model configurations
MODEL_TYPE = os.getenv("MODEL_TYPE", "stable-diffusion")
MODEL_ID = os.getenv("MODEL_ID", "CompVis/stable-diffusion-v1-4")
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", 7.5))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", 50))
HEIGHT = int(os.getenv("HEIGHT", 512))
WIDTH = int(os.getenv("WIDTH", 512))
PROMPT = os.getenv("PROMPT", "")
VARIANT = os.getenv("VARIANT", None)  # Optional variant parameter

# GPU and Memory configuration
USE_GPU = bool(strtobool(os.getenv("USE_GPU", "false")))
ENABLE_ATTENTION_SLICING = bool(strtobool(os.getenv("ENABLE_ATTENTION_SLICING", "true")))
ENABLE_MEMORY_EFFICIENT_ATTENTION = bool(strtobool(os.getenv("ENABLE_MEMORY_EFFICIENT_ATTENTION", "true")))
ENABLE_MODEL_CPU_OFFLOAD = bool(strtobool(os.getenv("ENABLE_MODEL_CPU_OFFLOAD", "false")))
ENABLE_SEQUENTIAL_CPU_OFFLOAD = bool(strtobool(os.getenv("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "false")))
EMPTY_CACHE_BETWEEN_RUNS = bool(strtobool(os.getenv("EMPTY_CACHE_BETWEEN_RUNS", "false")))
MAX_MEMORY = float(os.getenv("MAX_MEMORY", "1.0"))
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float32")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage statistics"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    logger.info(f"Memory Usage - RSS: {mem_info.rss / 1024 / 1024:.2f}MB, VMS: {mem_info.vms / 1024 / 1024:.2f}MB")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

class ImageGenerator:
    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        # Force CPU if USE_GPU is false
        if not USE_GPU:
            self.device = "cpu"
            torch.cuda.is_available = lambda: False  # Prevent CUDA checks when GPU is not requested
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"USE_GPU setting: {USE_GPU}")
        logger.info(f"Using device: {self.device}")

        if USE_GPU and self.device == "cpu":
            logger.warning("GPU was requested but is not available. Using CPU mode.")
        elif self.device == "cpu":
            logger.info("Running in CPU mode. This will be slower than GPU acceleration.")

        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Loading {MODEL_TYPE} model: {MODEL_ID}")

            # Determine model type and configuration
            if MODEL_TYPE == "PixArt-alpha":
                model_class = PixArtAlphaPipeline
            elif "Hyper-SD" in MODEL_ID:
                model_class = AutoPipelineForText2Image
            else:
                model_class = (
                    StableDiffusionXLPipeline if MODEL_TYPE == "stable-diffusion-xl"
                    else StableDiffusionPipeline
                )

            # Prepare model loading parameters
            model_params = {
                "use_safetensors": True
            }

            # Configure device-specific parameters
            if self.device == "cuda":
                # Set torch dtype based on configuration
                if TORCH_DTYPE == "float16":
                    model_params["torch_dtype"] = torch.float16
                    logger.info("Using FP16 precision")
                else:
                    model_params["torch_dtype"] = torch.float32
                    logger.info("Using FP32 precision")

                if VARIANT:  # Only add variant if it's specified
                    model_params["variant"] = VARIANT

                # Set maximum memory usage if specified
                if MAX_MEMORY < 1.0:
                    max_memory = {0: f"{MAX_MEMORY:.1%}"}
                    model_params["max_memory"] = max_memory
                    logger.info(f"Setting max GPU memory usage to {MAX_MEMORY:.1%}")

                logger.info(f"Loading model in GPU mode{' with variant ' + VARIANT if VARIANT else ''}")
            else:
                model_params["torch_dtype"] = torch.float32
                logger.info("Loading model in CPU mode with FP32")

            # Add offload parameters if enabled
            if ENABLE_MODEL_CPU_OFFLOAD:
                model_params["offload_folder"] = "offload"
                logger.info("Enabling model CPU offload")

            if ENABLE_SEQUENTIAL_CPU_OFFLOAD:
                model_params["device_map"] = "auto"
                logger.info("Enabling sequential CPU offload")

            # Load model with appropriate configuration
            self.model = model_class.from_pretrained(
                MODEL_ID,
                **model_params
            ).to(self.device)

            # Apply optimizations based on device and model
            if self.device == "cuda":
                # GPU-specific optimizations
                if ENABLE_ATTENTION_SLICING:
                    self.model.enable_attention_slicing()
                    logger.info("Enabled attention slicing for GPU")
                try:
                    self.model.enable_model_cpu_offload()
                    logger.info("Enabled CPU offload for GPU optimization")
                except Exception as e:
                    logger.warning(f"Could not enable GPU optimizations: {str(e)}")
            else:
                # CPU-specific optimizations
                if ENABLE_ATTENTION_SLICING:
                    self.model.enable_attention_slicing()
                    logger.info("Enabled attention slicing for CPU")
                if MODEL_TYPE == "stable-diffusion-xl":
                    self.model.enable_vae_slicing()
                    logger.info("Enabled VAE slicing for SDXL on CPU")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources and free memory"""
        if self.model is not None:
            try:
                # Move model to CPU before deletion
                if hasattr(self.model, 'to'):
                    self.model.to('cpu')

                # Delete model and clear CUDA cache
                del self.model
                self.model = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                logger.info("Model cleanup completed")
                log_memory_usage()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

    def generate(self, prompt=None):
        if prompt is None:
            prompt = PROMPT

        # Log memory usage before generation
        log_memory_usage()

        try:
            logger.info("Starting image generation")
            logger.info(f"Parameters: steps={NUM_INFERENCE_STEPS}, "
                      f"guidance_scale={GUIDANCE_SCALE}, "
                      f"dimensions={WIDTH}x{HEIGHT}")
            logger.info(f"Using device: {self.device}")

            # Ensure we're on CPU if CUDA is not available
            if not torch.cuda.is_available() and self.device == "cuda":
                logger.warning("CUDA not available, falling back to CPU...")
                self.device = "cpu"
                self.load_model()

            # Generate image with memory management
            try:
                image = self.model(
                    prompt=prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    height=HEIGHT,
                    width=WIDTH,
                ).images[0]
            finally:
                if EMPTY_CACHE_BETWEEN_RUNS and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache after generation")

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/image_{timestamp}_{str(uuid.uuid4())[:8]}.png"

            # Save image and cleanup
            image.save(filename)
            del image  # Explicitly delete the image
            logger.info(f"Image saved successfully: {filename}")

            # Log memory usage after generation
            log_memory_usage()
            return filename
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

# Initialize generator with memory monitoring
generator = ImageGenerator()
log_memory_usage()

@app.get("/")
async def root():
    return {"message": "Image Generation API is running"}

@app.post("/generate")
async def generate_image(custom_prompt: str = None):
    try:
        filename = generator.generate(custom_prompt)
        return {"message": "Image generated successfully", "filename": filename}
    except Exception as e:
        # Attempt cleanup on error
        generator.cleanup()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown"""
    logger.info("Shutting down application, cleaning up resources...")
    generator.cleanup()

if __name__ == "__main__":
    import uvicorn

    # Get number of workers from environment
    WORKERS = int(os.getenv("WORKERS", 1))

    # Performance Note:
    # Worker configuration impacts performance:
    # Single worker (default):
    # + Better memory efficiency (only one model loaded)
    # + Full GPU/CPU resources for each request
    # + Consistent generation times
    # - Only one request processed at a time
    #
    # Multiple workers:
    # + Can handle concurrent requests
    # - Splits GPU/CPU resources between workers
    # - Higher memory usage (model loaded per worker)
    # - May impact generation time per request
    #
    # For production scaling, consider:
    # 1. Multiple separate service instances (horizontal scaling)
    # 2. Queue system for handling multiple requests
    # 3. Load balancer to distribute requests across instances

    logger.info(f"Starting server with {WORKERS} worker{'s' if WORKERS > 1 else ''}")

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        workers=WORKERS
    )
