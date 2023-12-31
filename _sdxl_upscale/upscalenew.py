from diffusers import StableDiffusionUpscalePipeline as SD
from io import BytesIO
import torch, runpod, base64, os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
SD_UPSCALE_PATH = os.getenv("SD_UPSCALE_PATH")

pipe = SD.from_pretrained(SD_UPSCALE_PATH, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    image = job_input.get("image", None)
    if(image == None):
        return([])
    prompt = job_input.get("prompt", None)
    if(prompt == None):
        return([])
    steps = int(job_input.get("steps", 40))

    png = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")

    print("Generating Image(s)...")
    images = pipe(prompt=prompt, image=png, num_inference_steps=steps)

    send_image = []
    with BytesIO() as image_binary:
        images.save(image_binary, format="PNG")
        send_image.append(base64.b64encode(image_binary.getvalue()).decode)
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})