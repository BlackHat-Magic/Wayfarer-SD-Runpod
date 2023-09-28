import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline as SDXL
from diffusers import UniPCMultistepScheduler as Scheduler
from PIL import Image
from io import BytesIO

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

pipe = SDXL.from_pretrained(SDXL_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cat using a toaster.")
    negative_prompt = job_input.get("negative_prompt", "bad quality, worst quality, blurry, out of focus, cropped, out of frame, deformed, bad hands, bad anatomy")
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    steps = job_input.get("steps", 40)
    end_denoise = job_input.get("end_denoise", 0.8)
    guidance = job_input.get("guidance", 7.5)
    num_images = job_input.get("num_images", 4)

    # denoising_strength = job_input.get("denoising_strength", 0)
    # seed = job_input.get("seed", -1)
    # restore_faces = job_input["restore_faces"]
    # tiling = job_input["tiling"]
    # sampler_index = job_input["sampler_index"]

    print("Generating Image(s)...")
    unrefined = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        denoising_end=end_denoise,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        output_type="latent"
    ).images
    refined = refiner(
        prompt=prompt,
        num_inference_steps=steps,
        denoising_start=end_denoise,
        image=unrefined
    ).images

    send_image = []

    for image in refined:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        send_image.append(base64.b64encode(buffer.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})