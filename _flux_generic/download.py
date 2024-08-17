from diffusers import FluxPipeline as Flux
from huggingface_hub import login
from dotenv import load_dotenv
import torch, os, argparse

load_dotenv()
BASE_MODEL_PATH = os.getenv("FLUX_GENERIC_MODEL_PATH")

with open("./huggingfaceapi.key") as f:
    HUGGINGFACE_API_TOKEN = f.read()

login(HUGGINGFACE_API_TOKEN)

pipe = Flux.from_pretrained(
    BASE_MODEL_PATH,
    safety_checker=None,
    torch_dtype=torch.bfloat16,
)