import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionControlNetPipeline as CN
from diffusers import ControlNetModel, UniPCMultistepScheduler
from PIL import Image
from io import BytesIO

load_dotenv()
SD_MODEL_ID = os.getenv("SD_MODEL_ID")
SD_MODEL_OPENPOSE = os.getenv("SD_MODEL_OPENPOSE")
OPENPOSE_PORTRAIT = os.getenv("OPENPOSE_PORTRAIT")

controlnet_image = Image.open(OPENPOSE_PORTRAIT)
controlnet = ControlNetModel.from_single_file(SD_MODEL_OPENPOSE)
pipe = CN.from_single_file(SD_MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cat using a toaster.")
    negative_prompt = job_input.get("negative_prompt", "bad quality, worst quality, blurry, out of focus, cropped, out of frame, deformed, bad hands, bad anatomy")
    height = job_input.get("height", 512)
    width = job_input.get("width", 512)
    steps = job_input.get("steps", 30)
    guidance = job_input.get("guidance", 7.5)
    num_images = job_input.get("num_images", 4)

    # denoising_strength = job_input.get("denoising_strength", 0)
    # seed = job_input.get("seed", -1)
    # restore_faces = job_input["restore_faces"]
    # tiling = job_input["tiling"]
    # sampler_index = job_input["sampler_index"]

    print("Generating Image(s)...")
    images = pipe(
        prompt,
        controlnet_image,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images
    ).images

    send_image = []

    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        send_image.append(base64.b64encode(buffer.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})
