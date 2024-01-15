from diffusers import StableDiffusionUpscalePipeline as SD
from dotenv import load_dotenv
import torch, os

load_dotenv()
SD_UPSCALE_PATH = os.getenv("SD_UPSCALE_PATH")

pipe = SD.from_pretrained(
    SD_UPSCALE_PATH, 
    torch_dtype=torch.float16,
    vaiant="fp16",
    use_safetensors=True
)
