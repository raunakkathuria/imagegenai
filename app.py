import os
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import torch
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

# GPU configuration
USE_GPU = bool(strtobool(os.getenv("USE_GPU", "false")))
ENABLE_ATTENTION_SLICING = bool(strtobool(os.getenv("ENABLE_ATTENTION_SLICING", "true")))
ENABLE_MEMORY_EFFICIENT_ATTENTION = bool(strtobool(os.getenv("ENABLE_MEMORY_EFFICIENT_ATTENTION", "true")))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self):
        self.model = None
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
                model_params["torch_dtype"] = torch.float16
                if VARIANT:  # Only add variant if it's specified
                    model_params["variant"] = VARIANT
                logger.info(f"Loading model in GPU mode with FP16{' and variant ' + VARIANT if VARIANT else ''}")
            else:
                model_params["torch_dtype"] = torch.float32
                logger.info("Loading model in CPU mode with FP32")

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

    def generate(self, prompt=None):
        if prompt is None:
            prompt = PROMPT

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

            # Generate image
            image = self.model(
                prompt=prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=HEIGHT,
                width=WIDTH,
            ).images[0]

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/image_{timestamp}_{str(uuid.uuid4())[:8]}.png"

            # Save image
            image.save(filename)
            logger.info(f"Image saved successfully: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

# Initialize generator
generator = ImageGenerator()

@app.get("/")
async def root():
    return {"message": "Image Generation API is running"}

@app.post("/generate")
async def generate_image(custom_prompt: str = None):
    try:
        filename = generator.generate(custom_prompt)
        return {"message": "Image generated successfully", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8080)))
