__all__ = [
    "TextToImage",
    "SDXL",
    "SDXLImg2Img",
    "SDXLInpaint",
    "SD",
    "SDImg2Img",
    "SDInpaint",
    "Kadinsky",
    "KadinskyImg2Img",
    "KadinskyInpaint",
]

import json
import logging
import subprocess

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.TextToImage")
types = ["text2image", "img2img", "inpainting"]
ignore_for_types = {
    "text2image": ["image", "mask_image", "strength"],
    "img2img": ["mask_image"],
    "inpainting": [],
}


class ImageGen(BaseModel):
    def __init__(self):
        super().__init__()
        # image, mask_image (and some others) vary in
        # whether or not they're passed into the different pipes
        self.default_args = {
            "image": None,
            "mask_image": None,
            "prompt_2": None,
            "negative_prompt": None,
            "num_inference_steps": 40,
            "denoising_end": None,
            "strength": None,
            "guidance_scale": 7.0,
        }

    def load_model(self):
        raise NotImplementedError(
            "load_model() must be implemented in the derived class. Needs to set self.pipe, self.default_args and self.type"
        )

    def generate(self, prompt, **kwargs):
        prompt = prompt.strip()
        assert (
            prompt
        ), "Given empty prompt. Prompt needs to be passed and should be non-empty"
        if self.type == "img2img":
            assert (
                kwargs["image"] is not None
            ), "Please pass an initial PIL image for Img2Img generation"
        if self.type == "inpainting":
            assert (kwargs["image"] is not None) and (
                kwargs["mask_image"] is not None
            ), "Inpaiting needs both original image and mask images in PIL.Image format"

        args = dict()
        args["prompt"] = prompt
        if "image" not in ignore_for_types[self.type]:
            args["image"] = kwargs["image"]
        if "mask_image" not in ignore_for_types[self.type]:
            args["mask_image"] = kwargs["mask_image"]

        for k, v in self.default_args.items():
            if k in ignore_for_types[self.type]:
                continue
            else:
                if k in kwargs:
                    args[k] = kwargs[k]
                else:
                    args[k] = self.default_args[k]

        generated_image = self.pipe(**args).images[0]
        return generated_image


class SDXL(ImageGen):
    # for text to image on gpu
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline
    # Model card: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    # Pipelines: https://huggingface.co/docs/diffusers/using-diffusers/sdxl#stable-diffusion-xl

    # base="stabilityai/stable-diffusion-xl-base-1.0", 6.7GB
    def __init__(self, name="stabilityai/stable-diffusion-xl-base-1.0"):
        super().__init__()
        self.name = name
        self.type = types[0]
        self.default_args = {
            "prompt_2": None,
            "negative_prompt": None,
            "num_inference_steps": 40,
            "denoising_end": None,
            "guidance_scale": 7.0,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
            variant="fp16",
            use_safetensors=True,
        )  # .to("cuda")
        self.pipe.enable_model_cpu_offload()


class SDXLImg2Img(ImageGen):
    # for text+image to image on gpu
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline
    # Model card: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    # Pipelines: https://huggingface.co/docs/diffusers/using-diffusers/sdxl#stable-diffusion-xl

    # name="stabilityai/stable-diffusion-xl-refiner-1.0", 4.4GB
    def __init__(self, name="stabilityai/stable-diffusion-xl-refiner-1.0"):
        super().__init__()
        self.name = name
        self.type = types[1]
        self.default_args = {
            "prompt_2": None,
            "negative_prompt": None,
            "num_inference_steps": 40,
            "denoising_end": None,
            "strength": 0.75,
            "guidance_scale": 7.0,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.enable_model_cpu_offload()


class SDXLInpaint(ImageGen):
    # for inpainting on gpu
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline

    # name="stabilityai/stable-diffusion-xl-base-1.0", 6.7GB
    def __init__(self, name="stabilityai/stable-diffusion-xl-base-1.0"):
        super().__init__()
        self.name = name
        self.type = types[2]
        self.default_args = {
            "prompt_2": None,
            "negative_prompt": None,
            "num_inference_steps": 40,
            "denoising_end": None,
            "strength": 0.75,
            "guidance_scale": 7.0,
        }

        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.enable_model_cpu_offload()


class SD(ImageGen):
    # for text to image on cpu
    # Model card: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
    # See Limitations section

    # name="sd-legacy/stable-diffusion-v1-5", 5.2GB
    # with 20 steps ~15 min, 10 steps ~5 min, 5 steps ~2.5 min
    def __init__(self, name="sd-legacy/stable-diffusion-v1-5"):
        super().__init__()
        self.name = name
        self.type = types[0]
        self.default_args = {
            "prompt_2": None,
            "negative_prompt": None,
            "num_inference_steps": 20,
            "denoising_end": None,
            "guidance_scale": 7.0,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )  # .to(self.device)

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()


class SDImg2Img(ImageGen):
    # for text+image to image on cpu
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline

    # name="sd-legacy/stable-diffusion-v1-5", 5.2GB
    def __init__(self, name="sd-legacy/stable-diffusion-v1-5"):
        super().__init__()
        self.name = name
        self.type = types[1]
        self.default_args = {
            "prompt_2": None,
            "negative_prompt": None,
            "num_inference_steps": 20,
            "denoising_end": None,
            "strength": 0.75,
            "guidance_scale": 7.0,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()


class SDInpaint(ImageGen):
    # for inpainting on cpu
    # Model card: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting

    # ** init_image and mask_image should be PIL images.
    # ** The mask structure is white for inpainting and black for keeping as is

    # name="sd-legacy/stable-diffusion-inpainting", 5.2GB
    # with 10 steps ~3.5 min, 20steps ~7-10min
    def __init__(self, name="sd-legacy/stable-diffusion-inpainting"):
        super().__init__()
        self.name = name
        self.type = types[2]
        self.default_args = {
            "prompt_2": None,
            "negative_prompt": None,
            "num_inference_steps": 20,
            "denoising_end": None,
            "strength": 0.75,
            "guidance_scale": 7.0,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )  # .to(self.device)

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()


class Kadinsky(ImageGen):
    # for text to image on cpu
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/kandinsky#diffusers.KandinskyCombinedPipeline
    # Model card: https://huggingface.co/kandinsky-community/kandinsky-2-1

    # base="kandinsky-community/kandinsky-2-1"
    def __init__(self, name="kandinsky-community/kandinsky-2-1"):
        super().__init__()
        self.name = name
        self.type = types[0]
        self.default_args = {
            "negative_prompt": "low quality, bad quality",
            "num_inference_steps": 20,
            "guidance_scale": 4.0,
            "height": 512,
            "width": 512,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()


class KadinskyImg2Img(ImageGen):
    # for text+image to image on cpu
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/kandinsky#diffusers.KandinskyCombinedPipeline
    # Model card: https://huggingface.co/kandinsky-community/kandinsky-2-1

    # base="kandinsky-community/kandinsky-2-1"
    def __init__(self, name="kandinsky-community/kandinsky-2-1"):
        super().__init__()
        self.name = name
        self.type = types[1]
        self.default_args = {
            "negative_prompt": "low quality, bad quality",
            "num_inference_steps": 20,
            "guidance_scale": 4.0,
            "strength": 0.3,
            "height": 512,
            "width": 512,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt, **kwargs):
        h = kwargs.get("height", self.default_args["height"])
        w = kwargs.get("width", self.default_args["width"])
        kwargs["image"] = kwargs["image"].convert("RGB").thumbnail((h, w))

        generated_image = super().generate(prompt, **kwargs)
        return generated_image


class KadinskyInpaint(ImageGen):
    # for image inpainting on cpu
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/kandinsky#diffusers.KandinskyCombinedPipeline
    # Model card: https://huggingface.co/kandinsky-community/kandinsky-2-1

    # ** mask needs to be white
    # base="kandinsky-community/kandinsky-2-1-inpaint"
    def __init__(self, name="kandinsky-community/kandinsky-2-1-inpaint"):
        super().__init__()
        self.name = name
        self.type = types[2]
        self.default_args = {
            "negative_prompt": "low quality, bad quality",
            "num_inference_steps": 20,
            "guidance_scale": 4.0,
            "height": 512,
            "width": 512,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt, **kwargs):
        h = kwargs.get("height", self.default_args["height"])
        w = kwargs.get("width", self.default_args["width"])
        kwargs["image"] = kwargs["image"].convert("RGB").thumbnail((h, w))
        kwargs["mask_image"] = kwargs["mask_image"].convert("RGB").thumbnail((h, w))

        generated_image = super().generate(prompt, **kwargs)
        return generated_image


config = {
    "cpu": [
        {"model": SD, "name": "sd-legacy/stable-diffusion-v1-5"},
    ],
    "cuda": [
        {"model": SDXL, "name": "stabilityai/stable-diffusion-xl-base-1.0"},
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
