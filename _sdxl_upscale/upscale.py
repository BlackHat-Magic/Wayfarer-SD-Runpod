# from diffusers import StableDiffusionUpscalePipeline as SD
# from dotenv import load_dotenv
# from io import BytesIO
# from PIL import Image
# import runpod, os, torch, base64

# load_dotenv()
# SD_UPSCALE_PATH = os.getenv("SD_UPSCALE_PATH")

# pipe = SD.from_pretrained(SD_UPSCALE_PATH, torch_dtype=torch.float16).to("cuda")
# pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
# pipe.enable_xformers_memory_efficient_attention()

# def stable_diffusion(job):
#     job_input = job["input"]

#     image = job_input.get("image", None)
#     if(image == None):
#         return([])
#     prompt = job_input.get("prompt", None)

#     png = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
#     supersampled = png.resize((int(png.width * 2), int(png.height * 2)))

#     print("Generating Image(s)...")

#     refined = pipe(prompt=prompt, image=png)

#     send_image = []

#     with BytesIO() as image_binary:
#         refined.save(image_binary, format="PNG")
#         send_image.append(base64.b64encode(image_binary.getvalue()).decode())
    
#     return(send_image)

# runpod.serverless.start({"handler": stable_diffusion})

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from io import BytesIO
import base64, cv2, numpy, runpod

def esrgan(job):
    job_input = job["input"]
    scale = int(job_input.get("scale", 4))
    image = job_input.get("image", None)
    if(not image):
        return([])
    image = cv2.imdecode(numpy.frombuffer(base64.b64decode(image), numpy.uint8), cv2.IMREAD_COLOR)
    model = RRDBNet(
        num_in_ch=3, 
        num_out_ch=3, 
        num_feat=64, 
        num_block=23, 
        num_grow_ch=32, 
        scale=scale
    )
    upsampler = RealESRGANer(
        scale=scale, 
        model_path="./weights/RealESRGAN_x4plus.pth", 
        dni_weight=None, 
        model=model, 
        half=True
    )
    output, _ = upsampler.enhance(input)

    with BytesIO as image_binary:
        cv2.imwrite(image_binary, output)
        send_image = [base64.b64encode(image_binary.getvalue()).decode()]
    

    return(send_image)

runpod.serverless.start({"handler": esrgan})