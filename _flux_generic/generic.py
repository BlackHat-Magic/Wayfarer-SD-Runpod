import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import FluxPipeline as Flux
from huggingface_hub import login
from PIL import Image
from io import BytesIO

load_dotenv()
FLUX_MODEL_PATH = os.getenv("FLUX_MODEL_PATH")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
login(HUGGINGFACE_API_TOKEN)

pipe = Flux.from_pretrained(
    FLUX_MODEL_PATH, 
    torch_dtype=torch.float16, 
    use_safetensors=True
)
pipe.enable_model_cpu_offload()

def flux(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cat using a toaster.")
    # negative_prompt = job_input.get("negative_prompt", "bad quality, worst quality, blurry, out of focus, cropped, out of frame, deformed, bad hands, bad anatomy")
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    steps = job_input.get("steps", 50)
    guidance = job_input.get("guidance", 3.5)
    num_images = job_input.get("num_images", 4)

    # denoising_strength = job_input.get("denoising_strength", 0)
    # seed = job_input.get("seed", -1)
    # restore_faces = job_input["restore_faces"]
    # tiling = job_input["tiling"]
    # sampler_index = job_input["sampler_index"]

    print("Generating Image(s)...")
    images = pipe(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
    ).images

    send_image = []

    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        send_image.append(base64.b64encode(buffer.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": flux})