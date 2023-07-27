from torch import autocast
from diffusers import StableDiffusionPipeline as SD

def stable_diffusion(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cat using a toaster.")
    negative_prompt = job_input.get("negative_prompt", "bad quality, worst quality, blurry, out of focus, cropped, out of frame, deformed, bad hands, bad anatomy")
    height = job_input.get("height", 512)
    width = job_input.get("width", 512)
    steps = job_input.get("steps", 15)
    guidance = job_input.get("guidance", 7.5)
    num_images = job_input.get("num_images", 4)

    # denoising_strength = job_input.get("denoising_strength", 0)
    # seed = job_input.get("seed", -1)
    # restore_faces = job_input["restore_faces"]
    # tiling = job_input["tiling"]
    # sampler_index = job_input["sampler_index"]

    pipe = SD.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=wight,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images
    )
    for i, image in enumerate(images):
        image.save(f"image-{i}.png")

stable_diffusion({"input": {}})