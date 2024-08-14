from diffusers import StableDiffusionXLControlNetInpaintPipeline as SDXL
from diffusers import EulerAncestralDiscreteScheduler as Euler
from diffusers import ControlNetModel as CNM
from diffusers import DDIMScheduler as DDIM
from diffusers import AutoencoderKL as AKL
from PIL import Image, ImageDraw, ImageFilter
from botocore.client import Config
from dotenv import load_dotenv
import torch, io, base64, boto3, random, os, runpod

load_dotenv()

# load SDXL Model Stuff
SCHEDULER_MODEL_PATH = os.getenv("SDXL_UPSCALE_SCHEDULER_MODEL_PATH")
TILE_CONTROLNET_MODEL_PATH = os.getenv("SDXL_TILE_CONTROLNET_MODEL_PATH")
VAE_MODEL_PATH = os.getenv("SDXL_UPSCALE_VAE_MODEL_PATH")
BASE_MODEL_PATH = os.getenv("SDXL_UPSCALE_MODEL_PATH")

# Load S3 Credentials
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION_NAME = os.getenv("S3_REGION_NAME")

# initialize S3 session
session = boto3.session.Session()
s3_client = session.client(
    "s3",
    region_name=S3_REGION_NAME,
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4")
)

# function to generate random IDs for images
cuid = lambda x: "".join(random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-_") for _ in range(x))

# define pipeliens
scheduler = Euler.from_pretrained(
    SCHEDULER_MODEL_PATH,
    subfolder="scheduler"
)
controlnet = CNM.from_pretrained(
    CONTROLNET_MODEL_PATH,
    torch_dtype=torch.float16
)
vae = AKL.from_pretrained(
    VAE_MODEL_PATH,
    torch_dtype=torch.float16
)
pipe = SDXL.from_pretrained(
    BASE_UPSCALE_MODEL_PATH,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
    scheduler=scheduler
).to("cuda")

def stable_diffusion(job):
    # get job input
    job_input = job["input"]
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    steps = job_input.get("steps", 30)
    scale = job_input.get("scale", 4)
    guidance = job_input.get("guidance", 5.0)
    mask_blur = job_input.get("mask_blur", 48)

    # get image
    image_id = job_input.get("image_id", None)
    if(image_id == None):
        return("No image ID provided")
    try:
        s3_response = s3_client.get_object(Key=image_id, Bucket=S3_BUCKET_NAME)
    except Exception as e:
        print(e)
        return(f"Error: {e}")
    image_data = s3_response["Body"].read()
    image_binary = io.BytesIO(image_data)
    png = Image.open(image_binary)
    upsampled = png.resize((int(png.width * scale), int(png.height * scale)), Image.Resampling.LANCZOS)
    width, height = upsampled.size

    # create tiles and masks
    tile_width, tile_height = png.size
    upscaled_image = Image.new("RGB", (png.width * scale, png.height * scale))
    for i in range(scale):
        for j in range(scale):
            # intial crop points
            o_left = left = int(i * tile_width)
            o_top = top = int(j * tile_height)
            o_right = right = left + tile_width
            o_bottom = bottom = top + tile_height

            # mask boundaries
            b_left = b_top = 0

            # mask size
            mask_width = tile_width
            mask_height = tile_height

            # mask blur
            if(i > 0):
                top -= mask_blur
                b_top += mask_blur
                mask_height += mask_blur
            if(i < scale - 1):
                bottom += mask_blur
                mask_height += mask_blur
            if(j > 0):
                left -= mask_blur
                b_left += mask_blur
                mask_width += mask_blur
            if(j < scale - 1):
                right += mask_blur
                mask_width += mask_blur

            # mask boundaries again
            b_right = b_left + tile_width
            b_bottom = b_top + tile_height

            # create tile
            tile = upsampled.crop((left, top, right, bottom))

            # create mask
            mask = Image.new("RGB", (mask_width, mask_height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(mask)
            draw.rectangle([b_left, b_top, b_right, b_bottom], (255, 255, 255, 255))
            mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))

            # resample tile
            resampled = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance=guidance,
                image=tile,
                control_image=tile,
                mask_image=mask
            ).images[0]

            # paste
            cropped_resample = resampled.crop((b_left, b_top, b_right, b_bottom))
            upscaled_image.paste(cropped_resample, (o_left, o_top, o_right, o_bottom))
    
    # upload image
    try:
        image_binary = io.BytesIO()
        image.save(image_binary, format="png")
        image_binary.seek(0)
        key = cuid(8)
        s3_client.put_object(
            Key=f"{key}.png",
            Bucket=S3_BUCKET_NAME,
            Body=image_binary,
            ContentType="image/png",
            ACL="public-read"
        )
    except Exception as e:
        print(e)
        return(f"Error: {e}")
    
    # return image URL
    return(f"{S3_ENDPOINT_URL}/{S3_BUCKET_NAME}/{key}.png")

runpod.serverless.start({"handler": stable_diffusion})