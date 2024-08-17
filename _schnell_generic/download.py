from diffusers import FluxPipeline as Flux
from dotenv import load_dotenv
import torch, os

load_dotenv()
BASE_MODEL_FILENAME = os.getenv("FLUX_SCHNELL_FILENAME")

pipe = Flux.from_single_file(
    BASE_MODEL_FILENAME,
    safety_checker=None,
    torch_dtype=torch.bfloat16,
)