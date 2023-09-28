import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline as SDXL
from diffusers import DiffusionPipeline as Refiner
from diffusers import UniPCMultistepScheduler as Scheduler
from PIL import Image
from io import BytesIO

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")
SDXL_REFINER_PATH = os.getenv("SDXL_REFINER_PATH")

pipe = SDXL.from_pretrained(
    SDXL_MODEL_PATH, 
    torch_dtype=torch.float16, 
    variant = "fp16",
    use_safetensors=True
).to("cuda")
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

if(SDXL_REFINER_PATH != None != SDXL_REFINER_PATH != ""):
    refiner = Refiner.from_pretrained(
        SDXL_REFINER_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        # text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        # tokenizer=pipe.tokenizer,
        # tokenizer_2=pipe.tokenizer_2,
        # scheduler=pipe.scheduler
    ).to("cuda")
    refiner.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cat using a toaster.")
    negative_prompt = job_input.get("negative_prompt", "bad quality, worst quality, blurry, out of focus, cropped, out of frame, deformed, bad hands, bad anatomy")
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    steps = job_input.get("steps", 40)
    if(refiner):
        end_denoise = job_input.get("end_denoise", 0.8)
    else:
        end_denoise = 1.0
    guidance = job_input.get("guidance", 7.5)
    num_images = job_input.get("num_images", 4)

    # denoising_strength = job_input.get("denoising_strength", 0)
    # seed = job_input.get("seed", -1)
    # restore_faces = job_input["restore_faces"]
    # tiling = job_input["tiling"]
    # sampler_index = job_input["sampler_index"]

    print("Generating Image(s)...")
    if(refiner):
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
            num_images_per_prompt=num_images,
            image=unrefined
        ).images
    else:
        refined = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images,
        ).images

    send_image = []

    for image in refined:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        send_image.append(base64.b64encode(buffer.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})