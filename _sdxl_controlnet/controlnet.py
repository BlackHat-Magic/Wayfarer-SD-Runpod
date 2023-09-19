import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionXLControlNetPipeline as SDXL
from diffusers import DEISMultistepScheduler as Scheduler
from diffusers import ControlNetModel as CN
from PIL import Image
from io import BytesIO

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

# BLUR_CN_MODEL_PATH = os.getenv("BLUR_CN_MODEL_PATH")
CANNY_CN_MODEL_PATH = os.getenv("CANNY_CN_MODEL_PATH")
DEPTH_CN_MODEL_PATH = os.getenv("DEPTH_CN_MODEL_PATH")
# SCRIBBLE_CN_MODEL_PATH = os.getenv("SCRIBBLE_CN_MODEL_PATH")
OPENPOSE_CN_MODEL_PATH = os.getenv("OPENPOSE_CN_MODEL_PATH")
# REPLICATE_CN_MODEL_PATH = os.getenv("REPLICATE_CN_MODEL_PATH")

# BLUR_CN_STOP_STEP = int(os.getenv("BLUR_CN_STOP_STEP"))
CANNY_CN_STOP_STEP = int(os.getenv("CANNY_CN_STOP_STEP"))
DEPTH_CN_STOP_STEP = int(os.getenv("DEPTH_CN_STOP_STEP"))
# SCRIBBLE_CN_STOP_STEP = int(os.getenv("SCRIBBLE_CN_STOP_STEP"))
OPENPOSE_CN_STOP_STEP = int(os.getenv("OPENPOSE_CN_STOP_STEP"))
# REPLICATE_CN_STOP_STEP = int(os.getenv("REPLICATE_CN_STOP_STEP"))

# blur_controlnet = CN.from_single_file(BLUR_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
canny_controlnet = CN.from_pretrained(CANNY_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
depth_controlnet = CN.from_pretrained(DEPTH_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
# scribble_controlnet = CN.from_single_file(SCRIBBLE_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
openpose_controlnet = CN.from_pretrained(OPENPOSE_CN_MODEL_PATH, torch_dtype=torch.float16)
# replicate_controlnet = CN.from_single_file(REPLICATE_CN_MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)


def stable_diffusion(job):
    job_input = job["input"]

    image = job_input.get("images", None)
    image = Image.open(BytesIO(base64.b64decode(image[0]))).convert("RGB")
    prompt = job_input.get("prompt", None)
    if(prompt == None):
        return([])
    steps = int(job_input.get("steps", 30))
    model = job_input.get("model", None)
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    end_denoise = job_input.get("end_denoise", 1.0)
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

    pipe = SDXL.from_pretrained(SDXL_MODEL_PATH, torch_dtype=torch.float16, controlnet=controlnet)
    pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    print("Generating Image(s)...")
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        denoising_end=end_denoise,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        image = image,
        controlnet_conditioning_scale=1.0
    ).images

    send_image = []

    for image in images:
        with BytesIO() as image_binary:
            image.save(image_binary, format="PNG")
            send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})
