from diffusers import StableDiffusionXLPipeline as SDXL
from dotenv import load_dotenv
import torch, os

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")
# TILE_CN_MODEL_PATH = os.getenv("TILE_CN_MODEL_PATH")

# tile_controlnet = CN.from_pretrained(
#     TILE_CN_MODEL_PATH,
# )

pipe = SDXL.from_pretrained(
    SDXL_MODEL_PATH, 
    torch_dtype=torch.float16, 
    variant = "fp16",
    use_safetensors=True
)