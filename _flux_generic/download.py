from diffusers import FluxPipeline as Flux
from huggingface_hub import login
from dotenv import load_dotenv
import torch, os, argparse

load_dotenv()
BASE_MODEL_PATH = os.getenv("FLUX_GENERIC_MODEL_PATH")

with open("./huggingfaceapi.key") as f:
    # remove trailing whitespace because apparently nano and neovim are incapable of not adding a newline at the end for some goddamned reason
    HUGGINGFACE_API_TOKEN = str(f.read()).replace('\n', '')

login(HUGGINGFACE_API_TOKEN)

pipe = Flux.from_pretrained(
    BASE_MODEL_PATH,
    safety_checker=None,
    torch_dtype=torch.bfloat16,
)