import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionXLImg2ImgPipeline as SDXL
from diffusers import DiffusionPipeline as Refiner
from diffusers import UniPCMultistepScheduler as Scheduler
from diffusers.utils import load_image
from PIL import Image
from io import BytesIO

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

pipe = SDXL.from_pretrained(SDXL_MODEL_PATH, torch_dtype=torch.float16)
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
if(SDXL_REFINER_PATH != None and SDXL_REFINER_PATH != ""):
    refiner = Refiner.from_pretrained(
        SDXL_REFINER_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae
    ).to("cuda")
    refiner.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    image = job_input.get("image", None)
    if(image == None):
        return([])
    prompt = job_input.get("prompt", None)
    if(prompt == None):
        return([])
    steps = int(job_input.get("steps", 30))
    if(refiner):
        end_denoise = job_input.get("end_denoise", 0.8)
    else:
        end_denoise = 1.0

    png = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
    supersampled = png.resize((int(png.width * 2), int(png.height * 2)))

    print("Generating Image(s)...")
    if(refiner):
        unrefined = pipe(
            prompt=prompt,
            image=supersampled,
            num_inference_steps=steps,
            denoising_end=end_denoise,
            strength=0.4,
            output_type="latent"
        ).images[0]
        refined = refiner(
            prompt=prompt,
            num_inference_steps=steps,
            denoising_start=end_denoise,
            image=refined
        ).images[0]

    send_image = []

    with BytesIO() as image_binary:
        refined.save(image_binary, format="PNG")
        send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})
