from diffusers import StableDiffusionXLInpaintPipelineas Inpaint
from diffusers import StableDiffusionXLImg2ImgPipeline as SDXL
from diffusers import DPMSolverMultistepScheduler as Scheduler
from dotenv import load_dotenv
from PIL import Image
import base64, io, numpy, os, torch, runpod

load_dotenv()
SDXL_MODEL_PATH = os.getenv("SD_UPSCALE_PATH")

pipe = SDXL.from_pretrained(
    SDXL_MODEL_PATH, 
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)

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

    image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    upsampled = png.resize((int(png.width * scale), int(png.height * scale)), Image.LANCZOS)
    width, height = upsampled.size

    tile_width, tile_height = width, height
    while tile_width * tile_height > 1024 ** 2:
        tile_width //= 2
        tile_height //= 2

    print("Generating Image(s)...")

    # upsample each tile
    tiles = []
    padding = 32
    for i in range(0, width, tile_width):
        for j in range(0, height, tile_height):
            left_padding = 32
            left = i - padding
            if(left < 0):
                left = 0
                left_padding = 0
            
            top_padding = 32
            top = j - padding
            if(top < 0):
                top = 0
                top_padding = 0

            right_padding = 32
            right = i + tile_width + padding
            if(right > width):
                right = i + tile_width
                right_padding = 0

            bottom_padding = 32
            bottom = j + tile_height + padding
            if(bottom > height):
                bottom = j + tile_height
                bottom_padding = 0

            tile = upsampled.crop((left, top, right, bottom))
            og_width, og_height = tile.size
            
            tile = tile.resize((1024, 1024), Image.LANCZOS)

            with torch.no_grad():
                processed_tile = pipe(
                    prompt="",
                    num_inference_steps=steps,
                    image=tile,
                    guidance_scale=guidance,
                    strength=strength
                ).images[0]
            tiles.append({
                "left": left,
                "top": top,
                "left_padding": left_padding,
                "top_padding": top_padding,
                "og_width": og_width,
                "og_height": og_height,
                "image": processed_tile
            })

    # stitch them back together
    intermediate_image = Image.new("RGB", upsampled.size)
    for tile_obj in tiles:
        og_width = tile_obj["og_width"]
        og_height = tile_obj["og_height"]
        tile = tile_obj["image"].resize((width, height), Image.LANCZOS)
        
        left_padding = tile_obj["left_padding"]
        top_padding = tile_obj["top_padding"]
        tile = tile.crop((left_padding, top_padding, og_width + left_padding, og_height + top_padding))

        x = tile_obj["left"] + left_padding
        y = tile_obj["top"] + top_padding
        intermediate_image.paste(tile, (x, y))
    
    # switch to inpainting
    torch.cuda.empty_cache()
    pipe = Inpaint.from_pretrained(
        SDXL_MODEL_PATH, 
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)

    # generate vertical seam mask
    vertical_mask = Image.new("RGB", (tile_width, tile_height * 2))
    draw = ImageDraw.Draw(vertical_mask)
    draw.rectangle([tile_width // 4, 0, tile_width // 4 + tile_width // 2, tile_height * 2], (255, 255, 255, 255))
    vertical_mask = vertical_mask.filter(ImageFilter.GaussianBlur(128))
    vertical_mask = vertical_mask.crop((0, tile_height // 4, tile_width, tile_height // 4 + tile_height))

    # fix vertical seams
    fixed_vertical = intermediate_image.copy()
    for i in range(tile_width // 2, width, tile_width):
        for j in range(0, height, tile_height):
            left = i
            top = j
            right = i + tile_width
            bottom = j + tile_height

            tile = intermediate_image.crop((left, top, right, bottom))

            with torch.no_grad():
                processed_tile = inpaint_pipeline(
                    prompt="",
                    num_inference_steps=steps,
                    image=tile,
                    guidance_scale=guidance,
                    strength=strength,
                    mask_image=vertical_mask
                ).images[0]
            fixed_vertical.paste(processed_tile, (left, top))
    
    # generate horizontal masks
    first_horizontal_mask = Image.new("RGB", (tile_width * 2, tile_height))
    draw = ImageDraw.Draw(first_horizontal_mask)
    draw.rectangle([0, tile_height // 4, tile_width * 2, tile_height // 4 + tile_height // 2], (255, 255, 255, 0))
    first_horizontal_mask = first_horizontal_mask.filter(ImageFilter.GaussianBlur(128))
    first_horizontal_mask = first_horizontal_mask.crop((tile_width // 2, 0, tile_width + tile_width // 2, tile_height))

    subsequent_horizontal_mask = Image.new("RGB", (tile_width * 2, tile_height))
    draw = ImageDraw.Draw(subsequent_horizontal_mask)
    draw.rectangle([tile_width // 4, tile_height // 4, tile_width * 2, tile_height // 4 + tile_height // 2], (255, 255, 255, 0))
    subsequent_horizontal_mask = subsequent_horizontal_mask.filter(ImageFilter.GaussianBlur(128))
    subsequent_horizontal_mask = subsequent_horizontal_mask.crop((0, 0, tile_width, tile_height))

    subsequent_horizontal_mask

    # fix horizontal seams; making sure not to introduce new vertical ones.
    fixed_horizontal = fixed_vertical.copy()
    for i in range(0, width, tile_width // 2):
        for j in range(tile_height // 2, height, tile_height):
            left = i
            top = j
            right = i + tile_width
            bottom = j + tile_height

            tile = fixed_horizontal.crop((left, top, right, bottom))

            if(i == 0):
                horizontal_mask = first_horizontal_mask
            else:
                horizontal_mask = subsequent_horizontal_mask

            with torch.no_grad():
                processed_tile = inpaint_pipeline(
                    prompt="",
                    num_inference_steps=steps,
                    image=tile,
                    guidance_scale=guidance,
                    strength=strength,
                    mask_image=horizontal_mask
                ).images[0]
            fixed_horizontal.paste(processed_tile, (left, top))
    # only a short, simple, twelve-step process

    send_image = []

    with io.BytesIO() as image_binary:
        fixed_horizontal.save(image_binary, format="PNG")
        send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
    return(send_image)

runpod.serverless.start({"handler": stable_diffusion})