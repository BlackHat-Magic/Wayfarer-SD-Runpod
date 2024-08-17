from diffusers import StableDiffusionXLControlNetInpaintPipeline as SDXL
from diffusers import ControlNetModel as CNM
from diffusers import AutoencoderKL as AKL
from dotenv import load_dotenv
import torch, os

load_dotenv()
TILE_CONTROLNET_MODEL_PATH = os.getenv("SDXL_UPSCALE_TILE_CONTROLNET_MODEL_PATH")
VAE_MODEL_PATH = os.getenv("SDXL_UPSCALE_VAE_MODEL_PATH")
BASE_MODEL_PATH = os.getenv("SDXL_UPSCALE_MODEL_PATH")

controlnet = CNM.from_pretrained(
    TILE_CONTROLNET_MODEL_PATH,
    torch_dtype=torch.float16
)
vae = AKL.from_pretrained(
    VAE_MODEL_PATH,
    torch_dtype=torch.float16
)
pipe = SDXL.from_pretrained(
    BASE_MODEL_PATH,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
)