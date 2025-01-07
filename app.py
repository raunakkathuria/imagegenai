import os
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

class ImageGenerator:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        if MODEL_TYPE == "stable-diffusion":
            self.model = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        elif MODEL_TYPE == "stable-diffusion-xl":
            self.model = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

        self.model = self.model.to(self.device)
        # Enable memory efficient attention if using SDXL
        if MODEL_TYPE == "stable-diffusion-xl":
            self.model.enable_attention_slicing()

    def generate(self, prompt=None):
        if prompt is None:
            prompt = PROMPT

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
        return filename

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
