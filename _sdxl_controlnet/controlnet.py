import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionXLControlNetPipeline as SDXL
from diffusers import DiffusionPipeline as Refiner
from diffusers import UniPCMultistepScheduler as Scheduler
from diffusers import ControlNetModel as CN
from PIL import Image
from io import BytesIO

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")
SDXL_REFINER_PATH = os.getenv("SDXL_REFINER_PATH")

# BLUR_CN_MODEL_PATH = os.getenv("BLUR_CN_MODEL_PATH")
CANNY_CN_MODEL_PATH = os.getenv("CANNY_CN_MODEL_PATH")
DEPTH_CN_MODEL_PATH = os.getenv("DEPTH_CN_MODEL_PATH")
# SCRIBBLE_CN_MODEL_PATH = os.getenv("SCRIBBLE_CN_MODEL_PATH")
OPENPOSE_CN_MODEL_PATH = os.getenv("OPENPOSE_CN_MODEL_PATH")
# REPLICATE_CN_MODEL_PATH = os.getenv("REPLICATE_CN_MODEL_PATH")

# BLUR_CN_STOP_STEP = int(os.getenv("BLUR_CN_STOP_STEP"))
# CANNY_CN_STOP_STEP = int(os.getenv("CANNY_CN_STOP_STEP"))
# DEPTH_CN_STOP_STEP = int(os.getenv("DEPTH_CN_STOP_STEP"))
# SCRIBBLE_CN_STOP_STEP = int(os.getenv("SCRIBBLE_CN_STOP_STEP"))
# OPENPOSE_CN_STOP_STEP = int(os.getenv("OPENPOSE_CN_STOP_STEP"))
# REPLICATE_CN_STOP_STEP = int(os.getenv("REPLICATE_CN_STOP_STEP"))

# blur_controlnet = CN.from_single_file(BLUR_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
canny_controlnet = CN.from_pretrained(CANNY_CN_MODEL_PATH, torch_dtype=torch.float16)
depth_controlnet = CN.from_pretrained(DEPTH_CN_MODEL_PATH, torch_dtype=torch.float16)
# scribble_controlnet = CN.from_single_file(SCRIBBLE_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
openpose_controlnet = CN.from_pretrained(OPENPOSE_CN_MODEL_PATH, torch_dtype=torch.float16)
# replicate_controlnet = CN.from_single_file(REPLICATE_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)

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

    image = job_input.get("images", None)
    if(images == None):
        return([])
    image = Image.open(BytesIO(base64.b64decode(image[0]))).convert("RGB")
    prompt = job_input.get("prompt", None)
    negative_prompt = job_input.get("negative_prompt", "bad quality, worst quality, blurry, out of focus, cropped, out of frame, bad anatomy, bad hands, deformed")
    if(prompt == None):
        return([])
    steps = int(job_input.get("steps", 30))
    model = job_input.get("model", None)
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    if(refiner):
        end_denoise = job_input.get("end_denoise", 0.8)
    else:
        end_denoise = 1.0
    guidance = job_input.get("guidance", 7.5)
    num_images = job_input.get("num_images", 4)
    
    if(not model in ["canny", "depth", "openpose"]):
        print(model)
        print("nomodel")
        return([])
    controlnet = {
        "canny": canny_controlnet,
        "depth": depth_controlnet,
        "openpose": openpose_controlnet
    }[model]

    pipe = SDXL.from_pretrained(
        SDXL_MODEL_PATH, 
        torch_dtype=torch.float16, 
        controlnet=controlnet,
        use_safetensors=True,
        variant="fp16"
    )

    pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

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
            image = image,
            controlnet_conditioning_scale=1.0,
            output_type="latent"
        ).images
        refined = refiner(
            prompt=prompt,
            num_inference_steps=steps,
            denoising_start=end_denoise,
            num_images_per_prompt=num_images,
            image=unrefined
        )
    else:
        unrefined = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images,
            image = image,
            controlnet_conditioning_scale=1.0
        ).images

    send_image = []

    for image in refined:
        with BytesIO() as image_binary:
            image.save(image_binary, format="PNG")
            send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})
