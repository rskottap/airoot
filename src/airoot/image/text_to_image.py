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

import logging

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from airoot.base_model import BaseModel, try_load_models

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
    # for 20 steps ~15min,
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
        kwargs["image"] = kwargs["image"].convert("RGB")
        kwargs["image"].thumbnail((h, w))

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
        kwargs["image"] = kwargs["image"].convert("RGB")
        kwargs["mask_image"] = kwargs["mask_image"].convert("RGB")
        kwargs["image"].thumbnail((h, w))
        kwargs["mask_image"].thumbnail((h, w))

        generated_image = super().generate(prompt, **kwargs)
        return generated_image


class InstructPix2Pix(ImageGen):
    # for image editing
    # HF pipeline docs: https://huggingface.co/docs/diffusers/api/pipelines/pix2pix#diffusers.StableDiffusionInstructPix2PixPipeline

    # name="timbrooks/instruct-pix2pix", 4.4GB
    def __init__(self, name="timbrooks/instruct-pix2pix"):
        super().__init__()
        self.name = name
        self.type = types[1]
        self.default_args = {
            "negative_prompt": None,
            "num_inference_steps": 40,
            "strength": 0.75,
            "guidance_scale": 7.5,
        }
        self.load_model()

    def load_model(self):
        # load pipeline
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
        )
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt, **kwargs):
        kwargs["image"] = kwargs["image"].convert("RGB").resize((512, 512))
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


# Default
class TextToImage(BaseModel):
    def __new__(cls, module="TextToImage"):
        m = try_load_models(module)
        # ImageModel(name)
        self = m["model"](name=m["name"])
        return self
