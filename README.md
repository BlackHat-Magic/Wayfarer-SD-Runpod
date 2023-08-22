# Wayfarer-SD-Runpod

Runpod serverless code for the Stable Diffusion API used by Wayfarer and the Zoey Discord bot.

Uses [Huggingface Diffusers with ControlNet](https://huggingface.co/blog/controlnet)

Uses locally stored models instead of downloading from HuggingFace because `Lykon/DreamShaper` doesn't work with that method, and I really like that model.

Note: buildx's build context is meant to be the root directory of the project, as the `models/` and `img/` folders as well as the `requirements.txt` and `.env` files are shared between images are shared 