import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionXLImg2ImgPipeline as SDXL
from diffusers import UniPCMultistepScheduler as Scheduler
from diffusers.utils import load_image
from PIL import Image
from io import BytesIO

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

pipe = SDXL.from_single_file(SDXL_MODEL_PATH, torch_dtype=torch.float16)
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    image = job_input.get("image", None)
    if(image == None):
        return([])
    prompt = job_input.get("prompt", None)
    if(prompt == None):
        return([])
    steps = int(job_input.get("steps", 30))

    png = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
    supersampled = png.resize((int(png.width * 2), int(png.height * 2)))

    print("Generating Image(s)...")
    upscaled_image = pipe(
        prompt=prompt,
        image=supersampled,
        num_inference_steps=steps,
        strength=0.1
    ).images[0]

    send_image = []

    with BytesIO() as image_binary:
        upscaled_image.save(image_binary, format="PNG")
        send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})
