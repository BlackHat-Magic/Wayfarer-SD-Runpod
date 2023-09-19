from diffusers import StableDiffusionXLPipeline as SDXL
from diffusers import DEISMultiStepScheduler as Scheduler
from dotenv import load_dotenv

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

pipe = SDXL.from_pretrained(SDXL_MODEL_PATH, torch_dtype=torch.float16, controlnet=canny_controlnet)