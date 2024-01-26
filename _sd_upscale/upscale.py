from diffusers import StableDiffusionControlNetPipeline as SD
from diffusers import DDPMScheduler as Scheduler
from diffusers import ControlNetModel as CN
from dotenv import load_dotenv
from PIL import Image
import base64, io, numpy, os, torch, runpod

load_dotenv()
SD_UPSCALE_PATH = os.getenv("SD_UPSCALE_PATH")
TILE_CN_MODEL_PATH = os.getenv("TILE_CN_MODEL_PATH")

tile_controlnet = CN.from_pretrained(
    TILE_CN_MODEL_PATH,
).to("cuda")
tile_controlnet.enable_xformers_memory_efficient_attention()

pipe = SD.from_pretrained(
    SD_UPSCALE_PATH, 
    controlnet=[tile_controlnet],
    safety_checker=None
).to("cuda")
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    image = job_input.get("image", None)
    if(image == None):
        return([])
    prompt = job_input.get("prompt", None)
    negative_prompt = job_input.get("negative_prompt", None)
    steps = job_input.get("steps", 40)
    strength = job_input.get("strength", 1.0)
    scale = job_input.get("scale", 2.0)

    png = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    supersampled = png.resize((int(png.width * 2), int(png.height * 2)), Image.LANCZOS)
    width, height = supersampled.size

    print("Generating Image(s)...")

    refined = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        image=[supersampled],
        num_images_per_prompt=1,
        num_inference_steps=steps,
        strength=strength,
        width=width,
        height=height,
    ).images[0]

    send_image = []

    with io.BytesIO() as image_binary:
        refined.save(image_binary, format="PNG")
        send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})