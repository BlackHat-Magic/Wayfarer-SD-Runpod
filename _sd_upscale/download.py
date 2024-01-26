from diffusers import StableDiffusionControlNetPipeline as SD
from diffusers import ControlNetModel as CN
from dotenv import load_dotenv
import torch, os

load_dotenv()
SD_UPSCALE_PATH = os.getenv("SD_UPSCALE_PATH")
TILE_CN_MODEL_PATH = os.getenv("TILE_CN_MODEL_PATH")

tile_controlnet = CN.from_pretrained(
    TILE_CN_MODEL_PATH,
)

pipe = SD.from_pretrained(
    SD_UPSCALE_PATH, 
    controlnet=[tile_controlnet],
    safety_checker=None
)
