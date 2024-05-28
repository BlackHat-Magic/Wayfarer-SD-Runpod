from diffusers import StableDiffusionXLInpaintPipeline as SDXL
from diffusers import DPMSolverMultistepScheduler as Scheduler
from PIL import Image, ImageDraw, ImageFilter
from dotenv import load_dotenv
import requests, torch, base64, io, runpod, os

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SDXL_MODEL_PATH")

pipe = SDXL.from_pretrained(
    SDXL_MODEL_PATH,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)

def process_tile(pipe, tile: Image, steps: int, guidance: float, strength: float, mask: Image = None):
    with torch.no_grad():
        processed_tile = pipe(
            prompt="",
            num_inference_steps=steps,
            image=tile,
            guidance_scale=guidance,
            strength=strength,
            mask_image=mask
        ).images[0]
    return(processed_tile)

def initial_sd_resample(pipe, image, tile_width, tile_height, steps, guidance, strength):
    tiles = []
    resampled = image.copy()
    width, height = resampled.size
    for i in range(0, int(width // (4 / 3)), tile_width):
        for j in range(0, int(height // (4 / 3)), tile_height):
            left_padding = 32
            left = i - 32
            if(left < 0):
                left = 0
                left_padding = 0
            
            top_padding = 32
            top = j - 32
            if(top < 0):
                top = 0
                top_padding = 0

            right_padding = 32
            right = i + tile_width + 32
            if(right > width):
                right = width
                right_padding = 0

            bottom_padding = 32
            bottom = j + tile_height + 32
            if(bottom > height):
                bottom = height
                bottom_padding = 0

            mask_width = left_padding + tile_width + right_padding
            mask_height = top_padding + tile_height + bottom_padding
            mask = Image.new("RGB", (mask_width, mask_height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(mask)
            draw.rectangle([left_padding, top_padding, left_padding + tile_width, top_padding + tile_height], (255, 255, 255, 255))

            tile = resampled.crop((left, top, right, bottom))
            og_width, og_height = tile.size
            arr.append(tile.copy())
            
            tile = tile.resize((1024, 1024), Image.Resampling.LANCZOS)

            processed_tile = process_tile(pipe, tile, steps, guidance, strength, mask)

            # stitch back together
            processed_tile = processed_tile.resize((og_width, og_height), Image.Resampling.LANCZOS)
            processed_tile = processed_tile.crop((left_padding, top_padding, tile_width + left_padding, tile_height + top_padding))
            resampled.paste(processed_tile, (i, j))
    
    return(resampled)

def fix_seams(pipe, image, tile_width, tile_height, steps, guidance, strength, direction):
    seams_fix = image.copy()
    width, height = seams_fix.size
    if(direction == "h"):
        first_mask = Image.new("RGB", (tile_width * 2, tile_height))
        draw = ImageDraw.Draw(first_mask)
        draw.rectangle([0, tile_height // 4, tile_width * 2, tile_height - tile_height // 4], (255, 255, 255))
        first_mask = first_mask.filter(ImageFilter.GaussianBlur(tile_width // 8))
        first_mask = first_mask.crop((tile_width // 2, 0, tile_width + tile_width // 2, tile_height))
    
        subsequent_mask = Image.new("RGB", (tile_width * 2, tile_height))
        draw = ImageDraw.Draw(subsequent_mask)
        draw.rectangle([tile_width // 4, tile_height // 4, tile_width * 2, tile_height - tile_height // 4], (255, 255, 255))
        subsequent_mask = subsequent_mask.filter(ImageFilter.GaussianBlur(tile_width // 8))
        subsequent_mask = subsequent_mask.crop((0, 0, tile_width, tile_height))

        width_start = 0
        width_iter = int(tile_width * 0.75)
        height_start = tile_height // 2
    else:
        first_mask = Image.new("RGB", (tile_width, tile_height * 2))
        draw = ImageDraw.Draw(first_mask)
        draw.rectangle([tile_width // 4, 0, tile_width - tile_width // 4, tile_height * 2], (255, 255, 255))
        first_mask = first_mask.filter(ImageFilter.GaussianBlur(tile_width // 8))
        first_mask = first_mask.crop((0, tile_height // 4, tile_width, tile_height // 4 + tile_height))

        subsequent_mask = first_mask
        
        width_start = tile_width // 2
        width_iter = tile_width
        height_start = 0
    for i in range(width_start, int(width / (4 / 3) + 1), width_iter):
        for j in range(height_start, int(height // (4 / 3) + 1), tile_height):
            left = i
            top = j
            right = i + tile_width
            bottom = j + tile_height

            tile = seams_fix.crop((left, top, right, bottom))

            mask = subsequent_mask
            if(i == 0):
                mask = first_mask

            processed_tile = process_tile(pipe, tile, steps, guidance, strength, mask)
            seams_fix.paste(processed_tile, (left, top))
    
    return(seams_fix)

def stable_diffusion(job):
    job_input = job["input"]

    image = job_input.get("image", None)
    if(image == None):
        return([])
    prompt = job_input.get("prompt", None)
    negative_prompt = job_input.get("negative_prompt", None)
    steps = job_input.get("steps", 6)
    strength = job_input.get("strength", 0.5)
    scale = job_input.get("scale", 4.0)
    guidance = job_input.get("guidance", 2.0)

    png = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    upsampled = png.resize((int(png.width * scale), int(png.height * scale)), Image.LANCZOS)
    width, height = upsampled.size

    tile_width, tile_height = width, height
    while tile_width * tile_height > 1024 ** 2:
        tile_width //= 2
        tile_height //= 2

    print("Generating Image(s)...")

    first_pass = initial_sd_resample(pipe, upsampled, tile_width, tile_height, steps, guidance, strength)
    vertical = fix_seams(pipe, first_pass, tile_width, tile_height, steps, guidance, strength, "v")
    horizontal = fix_seams(pipe, first_pass, tile_width, tile_height, steps, guidance, strength, "v")

    with io.BytesIO() as image_binary:
        fixed_horizontal.save(image_binary, format="PNG")
        send_image = [base64.b64encode(image_binary.getvalue()).decode()]
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})