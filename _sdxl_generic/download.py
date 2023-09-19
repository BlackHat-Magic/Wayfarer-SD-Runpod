from diffusers import StableDiffusionXLPipeline as SDXL
from dotenv import load_dotenv
import torch, os

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

pipe = SDXL.from_pretrained(SDXL_MODEL_PATH, torch_dtype=torch.float16)