# Wayfarer-SD-Runpod

Runpod serverless code for the Stable Diffusion API used by Wayfarer and the Zoey Discord bot.

Uses [Huggingface Diffusers with ControlNet](https://huggingface.co/blog/controlnet)

Uses locally stored models instead of downloading from HuggingFace because `Lykon/DreamShaper` doesn't work with that method, and I really like that model.