from diffusers import FluxPipeline as Flux
from botocore.client import Config
from dotenv import load_dotenv
from PIL import Image
import base64, boto3, io, os, random, runpod, torch

load_dotenv()
BASE_MODEL_FILENAME = os.getenv("FLUX_SCHNELL_MODEL_PATH")

S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION_NAME = os.getenv("S3_REGION_NAME")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")

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

pipe = Flux.from_single_file(
    BASE_MODEL_FILENAME,
    safety_checker=None,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

def flux(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cat using a toaster.")
    # negative_prompt = job_input.get("negative_prompt", "bad quality, worst quality, blurry, out of focus, cropped, out of frame, deformed, bad hands, bad anatomy")
    height = job_input.get("height", 1024)
    width = job_input.get("width", 1024)
    steps = job_input.get("steps", 4)
    guidance = job_input.get("guidance", 0.0)
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
        max_sequence_length=256
        # clip_skip=-2
    ).images

    urls = []
    for image in images:
        image_binary = io.BytesIO()
        image.save(image_binary, format="PNG")
        image_binary.seek(0)
        key = cuid(8)
        try:
            s3_client.put_object(
                Key=f"{key}.png",
                Bucket=S3_BUCKET_NAME,
                Body=image_binary,
                ContentType="image/png",
                ACL="public-read"
            )
            urls.append(f"{S3_ENDPOINT_URL}/{S3_BUCKET_NAME}/{key}.png")
        except Exception as e:
            print(e)
            urls.append(f"Error: {e}")

    return(urls)

runpod.serverless.start({"handler": flux})