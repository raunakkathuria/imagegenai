import os
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
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
        # Check if GPU is requested and available
        self.device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"USE_GPU setting: {USE_GPU}")
        logger.info(f"Using device: {self.device}")

        if USE_GPU and not torch.cuda.is_available():
            logger.warning("GPU was requested but is not available. Falling back to CPU.")
        elif self.device == "cpu":
            logger.info("Running in CPU mode. This will be slower than GPU acceleration.")

        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Loading {MODEL_TYPE} model: {MODEL_ID}")

            # Determine model type and configuration
            model_class = (
                StableDiffusionXLPipeline if MODEL_TYPE == "stable-diffusion-xl"
                else StableDiffusionPipeline
            )

            def load_on_device(device, dtype):
                model = model_class.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    variant="fp16" if device == "cuda" else None,
                    use_safetensors=True,
                )
                model = model.to(device)
                return model

            try:
                # First attempt: Try loading with requested device
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = load_on_device(self.device, dtype)
                logger.info(f"Successfully loaded model on {self.device}")
            except (RuntimeError, Exception) as e:
                if self.device == "cuda":
                    # If GPU fails, fallback to CPU
                    logger.warning(f"Failed to load model on GPU: {str(e)}")
                    logger.info("Falling back to CPU...")
                    self.device = "cpu"
                    self.model = load_on_device("cpu", torch.float32)
                    logger.info("Successfully loaded model on CPU")
                else:
                    raise

            # Apply optimizations based on final device
            if ENABLE_ATTENTION_SLICING:
                self.model.enable_attention_slicing()
                logger.info("Enabled attention slicing")

            if self.device == "cuda":
                try:
                    self.model.enable_model_cpu_offload()
                    logger.info("Enabled CPU offload for GPU optimization")
                except Exception as e:
                    logger.warning(f"Could not enable GPU optimizations: {str(e)}")

            if ENABLE_MEMORY_EFFICIENT_ATTENTION:
                if MODEL_TYPE == "stable-diffusion-xl":
                    self.model.enable_vae_slicing()
                    logger.info("Enabled VAE slicing for SDXL")
                try:
                    self.model.enable_sequential_cpu_offload()
                    logger.info("Enabled sequential CPU offload")
                except Exception as e:
                    logger.warning(f"Could not enable CPU offload: {str(e)}")

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

            try:
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
            except RuntimeError as e:
                if "Found no NVIDIA driver" in str(e) and self.device == "cuda":
                    # If GPU fails during generation, try to reload model on CPU
                    logger.warning("GPU error during generation. Attempting CPU fallback...")
                    self.device = "cpu"
                    self.load_model()  # Reload model on CPU
                    return self.generate(prompt)  # Retry generation
                raise
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
