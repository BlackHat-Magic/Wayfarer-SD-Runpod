# Wayfarer-SD-Runpod

Runpod serverless code for the Stable Diffusion API used by Wayfarer and the Aphrodite Discord bot.

Uses [Huggingface Diffusers with ControlNet](https://huggingface.co/blog/controlnet)


Note: buildx's build context is meant to be the root directory of the project, as the `models/` and `img/` folders as well as the `requirements.txt` and `.env` files are shared between images are shared. Using the build scripts ensures this.
