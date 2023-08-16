import os, torch
from diffusers import StableDiffusionPipeline as SD
from diffusers import UniPCMultistepScheduler

load_dotenv()
SD_MODEL_ID = os.getenv("SD_MODEL_ID")

pipe = SD.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16)