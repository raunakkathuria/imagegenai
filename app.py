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
            logger.info(f"Loading {MODEL_TYPE} model: {MODEL_ID}")

            # Prepare model loading parameters
            model_params = {
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
                "device": self.device
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
                    memory_in_gb = int(torch.cuda.get_device_properties(0).total_memory * MAX_MEMORY / 1024 / 1024 / 1024)
                    max_memory = {0: f"{memory_in_gb}GB"}
                    model_params["max_memory"] = max_memory
                    logger.info(f"Setting max GPU memory to {memory_in_gb}GB")

                logger.info(f"Loading model in GPU mode{' with variant ' + VARIANT if VARIANT else ''}")
            else:
                model_params["torch_dtype"] = torch.float32
                logger.info("Loading model in CPU mode with FP32")

            # Add offload parameters if enabled
            if ENABLE_MODEL_CPU_OFFLOAD or ENABLE_SEQUENTIAL_CPU_OFFLOAD:
                model_params["device_map"] = "auto"
                logger.info("Enabling automatic device mapping for offload")
                # Remove direct device setting when using device_map
                model_params.pop("device", None)

            # Determine model type and configuration
            if MODEL_TYPE == "PixArt-alpha":
                model_class = PixArtAlphaPipeline
                # PixArt model settings
                if "LCM" in MODEL_ID:
                    logger.info("Detected LCM model variant, optimizing settings")
            elif MODEL_TYPE == "stable-diffusion-xl":
                model_class = StableDiffusionXLPipeline
                # SDXL specific optimizations
                if "turbo" in MODEL_ID.lower():
                    logger.info("Detected SDXL Turbo, optimizing settings")
                    if TORCH_DTYPE == "float16":
                        model_params["variant"] = "fp16"
                else:
                    logger.info("Loading SDXL base model")
                    model_params["vae_tiling"] = True
            elif "Hyper-SD" in MODEL_ID:
                model_class = AutoPipelineForText2Image
            else:
                model_class = StableDiffusionPipeline

            local_model_path = f"/app/models/{MODEL_TYPE}/{MODEL_ID.split('/')[-1]}"

            # Always load from HuggingFace for PixArt due to meta tensor issues
            if MODEL_TYPE == "PixArt-alpha":
                logger.info("Loading PixArt model directly from HuggingFace")
                self.model = model_class.from_pretrained(
                    MODEL_ID,
                    **model_params
                )
            else:
                if os.path.exists(local_model_path):
                    logger.info(f"Loading model from local cache: {local_model_path}")
                    # Load from local cache
                    self.model = model_class.from_pretrained(
                        local_model_path,
                        **model_params
                    )
                else:
                    # Download and save to local cache
                    logger.info("Model not found in cache. Starting download and pipeline loading:")
                    logger.info("1. Loading tokenizer and text encoder")
                    logger.info("2. Loading UNet for diffusion")
                    logger.info("3. Loading VAE for image encoding/decoding")
                    logger.info("4. Loading scheduler for inference steps")
                    logger.info("5. Finalizing pipeline setup")

                    self.model = model_class.from_pretrained(
                        MODEL_ID,
                        **model_params
                    )

                    # Save the model to local cache
                    os.makedirs(local_model_path, exist_ok=True)
                    logger.info(f"Saving model to local cache: {local_model_path}")
                    self.model.save_pretrained(local_model_path)

            logger.info("Pipeline components loaded successfully")

            # Apply optimizations based on device and model
            if self.device == "cuda":
                # GPU-specific optimizations
                if ENABLE_ATTENTION_SLICING:
                    self.model.enable_attention_slicing()
                    logger.info("Enabled attention slicing for GPU")
                try:
                    if MODEL_TYPE == "stable-diffusion-xl" and "turbo" not in MODEL_ID.lower():
                        # SDXL base model specific optimizations
                        self.model.enable_vae_tiling()
                        logger.info("Enabled VAE tiling for SDXL")
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

            # Generate image with memory management
            try:
                # Different pipelines have different callback parameter names
                callback_params = {}
                if hasattr(self.model, 'scheduler'):
                    logger.info("Setting up progress callback")

                    def callback_fn(step: int, _timestep: int, _latents: torch.FloatTensor) -> None:
                        logger.info(f"Generation progress: step {step + 1}/{NUM_INFERENCE_STEPS}")

                    # Try different parameter names used by different pipelines
                    if MODEL_TYPE == "PixArt-alpha":
                        callback_params["callback_on_step_end"] = callback_fn
                        callback_params["callback_on_step_end_tensor_inputs"] = ["latents"]
                    else:
                        callback_params["callback"] = callback_fn
                        callback_params["callback_steps"] = 1

                logger.info(f"Starting generation with {NUM_INFERENCE_STEPS} steps")
                image = self.model(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
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

@app.post("/generate")
async def generate_image(custom_prompt: str = None):
    try:
        if not generator:
            raise HTTPException(status_code=503, detail="Generator not initialized")
        filename = generator.generate(custom_prompt)
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
