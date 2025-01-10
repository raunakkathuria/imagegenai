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
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress FutureWarning from transformers
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Model configurations
MODEL_TYPE = os.getenv("MODEL_TYPE", "stable-diffusion")
MODEL_ID = os.getenv("MODEL_ID", "CompVis/stable-diffusion-v1-4")
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", 7.5))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", 50))
HEIGHT = int(os.getenv("HEIGHT", 512))
WIDTH = int(os.getenv("WIDTH", 512))
PROMPT = os.getenv("PROMPT", "")
NEGATIVE_PROMPT = os.getenv("NEGATIVE_PROMPT", "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, poorly drawn, extra limbs, bad proportions, gross proportions, text, watermark")
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
            logger.info(f"Loading model: {MODEL_ID}")

            # Configure base parameters
            model_params = {
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if TORCH_DTYPE == "float16" else torch.float32,
                "variant": VARIANT if VARIANT else None,
            }

            # Determine model class and configuration
            if MODEL_TYPE == "PixArt-alpha":
                model_class = PixArtAlphaPipeline
                logger.info("Loading PixArt model")
                # PixArt configuration
                model_params = {
                    "torch_dtype": torch.float16 if TORCH_DTYPE == "float16" else torch.float32,
                    "use_safetensors": True
                }
                if self.device == "cuda":
                    # For PixArt, we handle GPU placement after loading
                    logger.info("Will move PixArt model to GPU after loading")
            elif MODEL_TYPE == "stable-diffusion-xl":
                model_class = StableDiffusionXLPipeline
                logger.info("Loading SDXL model")
                if "turbo" in MODEL_ID.lower():
                    logger.info("Detected SDXL Turbo variant")
            else:
                # Use AutoPipeline as a fallback for better compatibility
                logger.info("Using AutoPipeline for model loading")
                model_class = AutoPipelineForText2Image

            # Load model
            model_name = MODEL_ID.split('/')[-1].replace('/', '_')
            local_model_path = f"/app/models/{model_name}"

            if os.path.exists(local_model_path):
                logger.info(f"Loading model from local cache: {local_model_path}")
                self.model = model_class.from_pretrained(
                    local_model_path,
                    **model_params
                )
            else:
                logger.info("Model not found in cache. Starting download and pipeline loading...")
                self.model = model_class.from_pretrained(
                    MODEL_ID,
                    **model_params
                )

                # Save the model to local cache
                os.makedirs(local_model_path, exist_ok=True)
                logger.info(f"Saving model to local cache: {local_model_path}")
                self.model.save_pretrained(local_model_path)

            logger.info("Pipeline components loaded successfully")

            # Move model to GPU if needed
            if self.device == "cuda" and not (ENABLE_MODEL_CPU_OFFLOAD or ENABLE_SEQUENTIAL_CPU_OFFLOAD):
                logger.info(f"Moving {MODEL_TYPE} model to GPU")
                self.model.to("cuda")
                # Verify GPU placement through the unet
                if hasattr(self.model, 'unet') and hasattr(self.model.unet, 'device'):
                    logger.info(f"Model device: {self.model.unet.device}")
                # Log GPU memory after model movement
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    logger.info(f"GPU Memory after model loading - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

            # Apply optimizations if not using offload
            if not (ENABLE_MODEL_CPU_OFFLOAD or ENABLE_SEQUENTIAL_CPU_OFFLOAD):
                if ENABLE_ATTENTION_SLICING:
                    self.model.enable_attention_slicing()
                    logger.info("Enabled attention slicing")

                if hasattr(self.model, 'enable_vae_tiling'):
                    self.model.enable_vae_tiling()
                    logger.info("Enabled VAE tiling")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources and free memory"""
        if self.model is not None:
            try:
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

    def generate(self, prompt=None, negative_prompt=None):
        if prompt is None:
            prompt = PROMPT
        if negative_prompt is None:
            negative_prompt = NEGATIVE_PROMPT

        # Log memory usage before generation
        log_memory_usage()

        try:
            logger.info("Starting image generation")
            logger.info(f"Parameters: steps={NUM_INFERENCE_STEPS}, "
                      f"guidance_scale={GUIDANCE_SCALE}, "
                      f"dimensions={WIDTH}x{HEIGHT}")
            logger.info(f"Using device: {self.device}")

            # Generate image with memory management
            try:
                # Configure callback for progress tracking
                callback_params = {}
                if hasattr(self.model, 'scheduler'):
                    logger.info("Setting up progress callback")

                    def callback_fn(step: int, _timestep: int, _latents: torch.FloatTensor) -> None:
                        logger.info(f"Generation progress: step {step + 1}/{NUM_INFERENCE_STEPS}")

                    callback_params["callback"] = callback_fn
                    callback_params["callback_steps"] = 1

                logger.info(f"Starting generation with {NUM_INFERENCE_STEPS} steps")
                image = self.model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    height=HEIGHT,
                    width=WIDTH,
                    **callback_params
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

# Global generator instance
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    global generator
    # Startup: Initialize generator
    generator = ImageGenerator()
    log_memory_usage()
    yield
    # Shutdown: cleanup resources
    if generator:
        logger.info("Shutting down application, cleaning up resources...")
        generator.cleanup()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Image Generation API is running"}

from pydantic import BaseModel

class GenerateRequest(BaseModel):
    custom_prompt: str | None = None
    negative_prompt: str | None = None

@app.post("/generate")
async def generate_image(request: GenerateRequest | None = None):
    try:
        if not generator:
            raise HTTPException(status_code=503, detail="Generator not initialized")

        # Use custom prompts if provided, otherwise fall back to environment variables
        prompt = request.custom_prompt if request and request.custom_prompt is not None else PROMPT
        negative = request.negative_prompt if request and request.negative_prompt is not None else NEGATIVE_PROMPT

        logger.info(f"Using prompt: {prompt}")
        logger.info(f"Using negative prompt: {negative}")

        filename = generator.generate(prompt, negative)
        return {"message": "Image generated successfully", "filename": filename}
    except Exception as e:
        # Attempt cleanup on error
        if generator:
            generator.cleanup()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import sys
    import pathlib

    # Get number of workers from environment
    WORKERS = int(os.getenv("WORKERS", 1))

    if WORKERS > 1:
        logger.warning("Using multiple workers may cause high memory usage and potential instability")
        logger.warning("Consider using WORKERS=1 for better stability")

    # Get the directory containing the script
    file_path = pathlib.Path(__file__).parent.absolute()
    sys.path.append(str(file_path))

    uvicorn.run(
        "app:app",  # Use module:app format
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        workers=WORKERS,
        reload=False  # Disable reload to prevent duplicate model loading
    )
