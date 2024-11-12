__all__ = [
    "TextToImage",
    "SDXLBase",
    "SDXLRefiner",
    "SDXLFull",
]

import json
import logging
import subprocess

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.TextToImage")


class SDXLBase(BaseModel):
    # for text to image on gpu
    # HF docs: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline
    # Model card: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

    # base="stabilityai/stable-diffusion-xl-base-1.0", 6.7GB
    def __init__(self, name="stabilityai/stable-diffusion-xl-base-1.0"):
        super().__init__()
        self.base_name = name
        self.load_model()

    def load_model(self):
        # load base model
        self.base = DiffusionPipeline.from_pretrained(
            self.base_name,
            torch_dtype=self.torch_dtype,
            variant="fp16",
            use_safetensors=True,
        )  # .to(self.device)

        # When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.base.enable_model_cpu_offload()

    def generate(
        self,
        prompt,
        prompt_2=None,
        negative_prompt=None,
        n_steps=40,
        denoising_end=0.8,
        target_size=(1024, 1024),
        output_type="latent",
    ):
        # returns Tensor object
        base_image = self.base(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=denoising_end,
            output_type=output_type,
            original_size=target_size,
            target_size=target_size,
        ).images
        return base_image

    def convert_to_pil(base_image):
        import torchvision.transforms as transforms

        to_pil_image = transforms.ToPILImage()
        pil_image = to_pil_image(base_image)
        return pil_image


class SDXLRefiner(BaseModel):
    # for text to image on gpu
    # HF docs: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline
    # Model card: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

    # refiner="stabilityai/stable-diffusion-xl-refiner-1.0", 4.4GB
    def __init__(self, name="stabilityai/stable-diffusion-xl-refiner-1.0"):
        super().__init__()
        self.refiner_name = "stabilityai/stable-diffusion-xl-refiner-1.0"
        base = SDXLBase()
        self.text_encoder_2 = base.base.text_encoder_2
        self.vae = base.base.vae
        del base
        self.load_model()

    def load_model(self):
        self.refiner = DiffusionPipeline.from_pretrained(
            self.refiner_name,
            text_encoder_2=self.text_encoder_2,
            vae=self.vae,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16",
        )  # .to(self.device)

        # When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile.
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)
        self.refiner.enable_model_cpu_offload()

    def generate(
        self,
        guide_image,
        prompt,
        prompt_2=None,
        negative_prompt=None,
        n_steps=40,
        denoising_end=0.8,
        target_size=(1024, 1024),
    ):
        # returns PIL.Image.Image type
        refined_image = self.refiner(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=denoising_end,
            original_size=target_size,
            target_size=target_size,
            image=guide_image,
        ).images[0]
        return refined_image


class SDXLFull(BaseModel):
    def __init__(self, name="full"):
        super().__init__()
        self.load_model()

    def load_model(self):
        # load base model
        self.base = SDXLBase()
        self.refiner = SDXLRefiner()
        self.base_name = self.base.base_name
        self.refiner_name = self.refiner.refiner_name

    def generate(
        self,
        prompt,
        prompt_2=None,
        negative_prompt=None,
        n_steps=40,
        denoising_end=0.8,
        target_size=(1024, 1024),
    ):
        # returns PIL.Image.Image type
        base_image = self.base.generate(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            n_steps=n_steps,
            denoising_end=denoising_end,
            output_type="latent",
            target_size=target_size,
        )

        refined_image = self.refiner.generate(
            guide_image=base_image,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            n_steps=n_steps,
            denoising_end=denoising_end,
            target_size=target_size,
        )
        return refined_image


config = {
    "cpu": [
        {"model": SDXLFull, "name": "full"},
    ],
    "cuda": [
        {"model": SDXLFull, "name": "full"},
    ],
}


def try_load_models(module) -> dict:
    """
    Tries to load the model into memory, in order of devices.

    Does this by trying to load the model into memory in a separate process. So if it fails mid-way, with some model layers loaded into memory but not all, and raises an exception, the GPU memory gets cleared up automatically when the process exits. Otherwise, we won't have access to the model class/variable to be able to delete it later and clear up memory. Hence, trying it in a different process.

    If successful, writes that model to etc.cache_path as the default model to use on that machine for the module [audio, video, image, text].

    If gpu is available then:
        - uses gpu regardless of model.
        - first try to load the cuda models in order (decreasing memory usage).
        - If all the cuda models fail, then switch to cpu default models but fit them to gpu.

    If no gpu is available then try to load the cpu models in order.
    """
    model = get_default_model(module)
    if model:
        return model
    logger.info(
        f"No default model found (in ~/.cache/airoot) for {module} on this device. Trying to determine default model for this device."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # order in which to try out the models
    params = []

    # first try to load cuda models
    if device == "cuda":
        defaults = config["cuda"]
        params.extend(["--keys", "cuda", "--idx", i] for i in range(len(defaults)))
    # either device is cpu or cuda models failed to load
    defaults = config["cpu"]
    params.extend(["--keys", "cpu", "--idx", i] for i in range(len(defaults)))

    import os

    import airoot

    for p in params:
        try:
            model = config[p[1]][p[3]]
            p = [str(x) for x in p]
            test_path = os.path.join(airoot.__path__[-1], "test_load_model.py")
            command = [
                "python3",
                test_path,
                "-m",
                module,
            ] + p
            _ = subprocess.run(
                command,
                capture_output=True,
                check=True,
            )
            set_default_model([p[1], p[3]], module, model)
            return model
        except Exception as e:
            logger.error(
                f"Unable to load model {model['name']} on {device.upper()} due to error:\n{e.stderr}\nTrying next model.",
                exc_info=True,
            )
            continue

    raise Exception(
        f"Unable to load any of the default models for {module} module. All available models:\n{json.dumps(config, default=lambda o: str(o), indent=2)}"
    )


# Default
class TextToImage(BaseModel):
    def __new__(cls, module="TextToImage"):
        m = try_load_models(module)
        # ImageModel(name)
        self = m["model"](name=m["name"])
        return self
