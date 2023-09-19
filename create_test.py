from PIL import Image
import io, json, base64

test_input = {
    "prompt": "Drab, gloomy, desaturated, depressing, hopeless digital painting portrait of a mid-20s adventurer in with bright blonde hair and vibrant blue eyes, with a scar over his nose, face scar, scar on face medieval setting"
}

images = []

with Image.open("./preprocessed.png") as image:
    image_binary = io.BytesIO()
    image.save(image_binary, "PNG")
    images.append(base64.b64encode(image_binary.getvalue()).decode("utf-8"))

test_input["images"] = images
wrapper = {"input": test_input}

with open("test_input.json", "w") as json_file:
    json.dump(wrapper, json_file)