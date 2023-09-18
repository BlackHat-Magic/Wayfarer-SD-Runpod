from dotenv import load_dotenv
from diffusers import StableDiffusionUpscalePipeline as Upscale
import torch, os

load_dotenv()
UPSCALE_MODEL = os.getenv("UPSCALE_MODEL")

pipe = Upscale.from_pretrained(UPSCALE_MODEL, torch_dtype=torch.float16)
