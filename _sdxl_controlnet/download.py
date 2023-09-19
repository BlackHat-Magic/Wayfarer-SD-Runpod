from diffusers import StableDiffusionXLControlNetPipeline as SDXL
from diffusers import ControlNetModel as CN
from dotenv import load_dotenv
import torch, os

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")
CANNY_CN_MODEL_PATH = os.getenv("CANNY_CN_MODEL_PATH")
DEPTH_CN_MODEL_PATH = os.getenv("DEPTH_CN_MODEL_PATH")
OPENPOSE_CN_MODEL_PATH = os.getenv("OPENPOSE_CN_MODEL_PATH")

canny_controlnet = CN.from_pretrained(CANNY_CN_MODEL_PATH, torch_dtype=torch.float16)
depth_controlnet = CN.from_pretrained(DEPTH_CN_MODEL_PATH, torch_dtype=torch.float16)
openpose_controlnet = CN.from_pretrained(OPENPOSE_CN_MODEL_PATH, torch_dtype=torch.float16)

pipe = SDXL.from_pretrained(SDXL_MODEL_PATH, torch_dtype=torch.float16, controlnet=canny_controlnet)