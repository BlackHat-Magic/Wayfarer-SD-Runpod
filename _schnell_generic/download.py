from diffusers import FluxPipeline as Flux
from dotenv import load_dotenv
import torch, os

load_dotenv()
BASE_MODEL_PATH = os.getenv("FLUX_SCHNELL_MODEL_PATH")

pipe = Flux.from_pretrained(
    BASE_MODEL_PATH,
    safety_checker=None,
    torch_dtype=torch.bfloat16,
)