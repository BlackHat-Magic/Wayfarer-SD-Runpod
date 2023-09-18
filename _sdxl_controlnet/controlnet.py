import runpod, os, torch, base64
from dotenv import load_dotenv
from diffusers import StableDiffusionXLControlNetPipeline as SDXL
from diffusers import UniPCMultistepScheduler as Scheduler
from diffusers import ControlNetModel as CN
from PIL import Image
from io import BytesIO

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

BLUR_CN_MODEL_PATH = os.getenv("BLUR_CN_MODEL_PATH")
CANNY_CN_MODEL_PATH = os.getenv("CANNY_CN_MODEL_PATH")
DEPTH_CN_MODEL_PATH = os.getenv("DEPTH_CN_MODEL_PATH")
SCRIBBLE_CN_MODEL_PATH = os.getenv("SCRIBBLE_CN_MODEL_PATH")
OPENPOSE_CN_MODEL_PATH = os.getenv("OPENPOSE_CN_MODEL_PATH")
REPLICATE_CN_MODEL_PATH = os.getenv("REPLICATE_CN_MODEL_PATH")

BLUR_CN_STOP_STEP = int(os.getenv("BLUR_CN_STOP_STEP"))
CANNY_CN_STOP_STEP = int(os.getenv("CANNY_CN_STOP_STEP"))
DEPTH_CN_STOP_STEP = int(os.getenv("DEPTH_CN_STOP_STEP"))
SCRIBBLE_CN_STOP_STEP = int(os.getenv("SCRIBBLE_CN_STOP_STEP"))
OPENPOSE_CN_STOP_STEP = int(os.getenv("OPENPOSE_CN_STOP_STEP"))
REPLICATE_CN_STOP_STEP = int(os.getenv("REPLICATE_CN_STOP_STEP"))

blur_controlnet = CN.from_single_file(BLUR_CN_STOP_STEP, torch_dtype=torch.float16)
canny_controlnet = CN.from_single_file(CANNY_CN_STOP_STEP, torch_dtype=torch.float16)
depth_controlnet = CN.from_single_file(DEPTH_CN_STOP_STEP, torch_dtype=torch.float16)
scribble_controlnet = CN.from_single_file(SCRIBBLE_CN_STOP_STEP, torch_dtype=torch.float16)
openpose_controlnet = CN.from_single_file(OPENPOSE_CN_STOP_STEP, torch_dtype=torch.float16)
replicate_controlnet = CN.from_single_file(REPLICATE_CN_STOP_STEP, torch_dtype=torch.float16)

pipe = SDXL.from_single_file(SDXL_MODEL_PATH, torch_dtype=torch.float16)
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

def stable_diffusion(job):
    job_input = job["input"]

    images = job_input.get("images", None)
    if(images == None or images == []):
        return([])
    prompt = job_input.get("prompt", None)
    if(prompt == None):
        return([])
    steps = int(job_input.get("steps", 30))
    model = job_input.get("model", None)
    if(model == None):
        return([])
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    end_denoise = job_input.get("end_denoise", 1.0)
    guidance = job_input.get("guidance", 7.5)
    num_images = job_input.get("num_images", 4)

    loaded_images = []
    controlnets = []
    for key, value in images.items():
        if(value == None):
            continue
        if(key not in ["blur", "canny", "depth", "scribble", "openpose", "replicate"]):
            continue
        loaded_images.append(Image.open(BytesIO(base64.b64decode(value)).convert("RGB")))
        match key:
            case "blur":
                controlnets.append(blur_controlnet)
            case "canny":
                controlnets.append(canny_controlnet)
            case "depth":
                controlnets.append(depth_controlnet)
            case "scribble":
                controlnets.append(scribble_controlnet)
            case "openpose":
                controlnets.append(openpose_controlnet)
            case "replicate":
                controlnets.append(replicate_controlnet)
    
    pipe.controlnets = controlnets

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
        image = loaded_images,
        controlnet_conditioning_scale=[min(1.0, 2/len(controlnets)) for cn in controlnets]
    ).images

    send_image = []

    for image in images:
        with BytesIO() as image_binary:
            image.save(image_binary, format="PNG")
            send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})