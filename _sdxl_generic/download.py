from diffusers import StableDiffusionXLPipeline as SDXL
from dotenv import load_dotenv
import torch, os

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")
SDXL_REFINER_PATH = os.getenv("SDXL_REFINER_PATH")

pipe = SDXL.from_pretrained(
    SDXL_MODEL_PATH, 
    torch_dtype=torch.float16, 
    variant = "fp16",
    use_safetensors=True
)
if(SDXL_REFINER_PATH != None and SDXL_REFINER_PATH != ""):
    refiner = Refiner.from_pretrained(
        SDXL_REFINER_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae
    )