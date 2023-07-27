import runpod, os, torch, numpy
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline as SD
from diffusers import StableDiffusionControlNetPipeline as CN
from diffusers import ControlNetModel, UniPCMultistepScheduler
from PIL import Image

load_dotenv()
CHARACTER_MODEL_PATH = os.getenv("CHARACTER_MODEL_PATH")
SD_MODEL_OPENFACE = os.getenv("SD_MODEL_OPENFACE")
OPENFACE_REFERENCE = os.getenv("OPENFACE_REFERENCE")

# pipe = SD.from_single_file(CHARACTER_MODEL_PATH, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

openface_image = Image.open(OPENFACE_REFERENCE)
openface_controlnet = ControlNetModel.from_single_file(SD_MODEL_OPENFACE)
openface_pipe = CN.from_single_file(CHARACTER_MODEL_PATH, controlnet=openface_controlnet, torch_dtype=torch.float16)
openface_pipe.scheduler = UniPCMultistepScheduler.from_config(openface_pipe.scheduler.config)
openface_pipe.enable_model_cpu_offload()
openface_pipe.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A goblin man.")
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

    # images = pipe(
    #     prompt,
    #     negative_prompt=negative_prompt,
    #     height=height,
    #     width=width,
    #     num_inference_steps=steps,
    #     guidance_scale=guidance,
    #     num_images_per_prompt=num_images
    # ).images

    images = openface_pipe(
        prompt,
        openface_image,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images
    ).images
    
    return(images)

runpod.serverless.start({"handler": stable_diffusion})
