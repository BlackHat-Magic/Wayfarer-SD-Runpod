import os, torch
from diffusers import StableDiffusionControlNetPipeline as CN
from diffusers import ControlNetModel, UniPCMultistepScheduler

load_dotenv()
SD_MODEL_ID = os.getenv("SD_MODEL_ID")
SD_MODEL_OPENPOSE = os.getenv("SD_MODEL_OPENPOSE")

controlnet = ControlNetModel.from_pretrained(SD_MODEL_OPENPOSE)
pipe = CN.from_pretrained(SD_MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16)