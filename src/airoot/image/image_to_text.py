__all__ = [
    "ImageToText",
    "Blip",
    "Blip2",
    "EasyOCR",
    "LlavaNext",
    "Florence",
]

import json
import logging
import subprocess
import textwrap

import easyocr
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.ImageToText")


class Blip(BaseModel):
    # for image to text on cpu
    # link: https://huggingface.co/Salesforce/blip-image-captioning-large
    # name="Salesforce/blip-image-captioning-large"
    # link: https://huggingface.co/Salesforce/blip-vqa-base
    # name="Salesforce/blip-vqa-base" # For VQA

    def __init__(self, name="Salesforce/blip-image-captioning-large"):
        super().__init__()
        self.name = name
        self.default_prompt = "Describe this image in detail."
        self.default_prompt = "What is this image about?"
        self.load_model()

    def load_model(self):
        self.processor = BlipProcessor.from_pretrained(self.name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=512):
        # NOTE: blip-image-captioning-large and vqa-base don't do well with prompts
        image = image_data
        if self.name == "Salesforce/blip-image-captioning-large":
            inputs = self.processor(image, return_tensors="pt").to(
                self.device, self.torch_dtype
            )
        else:
            inputs = self.processor(image, text=text, return_tensors="pt").to(
                self.device, self.torch_dtype
            )
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        generated_text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return generated_text


class Blip2(BaseModel):
    # link: https://huggingface.co/Salesforce/blip2-opt-2.7b
    # name="Salesforce/blip2-opt-2.7b" # For BLIP2

    def __init__(self, name="Salesforce/blip2-opt-2.7b"):
        super().__init__()
        self.name = name
        self.template = "Question: {} Answer:"
        self.default_prompt = "What is this image about?"
        self.load_model()

    def load_model(self):
        self.processor = Blip2Processor.from_pretrained(self.name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=512):
        if text is None:
            text = self.template.format(self.default_prompt)
        else:
            text = self.template.format(text)

        inputs = self.processor(image_data, text=text, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return generated_text


class EasyOCR(BaseModel):
    # for image ocr on cpu/gpu
    # link: https://github.com/JaidedAI/EasyOCR
    """
    EasyOCR has some limits, so wrap generate in a try except.
    - File extension support: png, jpg, tiff.
    - File size limit: 2 Mb.
    - Image dimension limit: 1500 pixel.
    - Possible Language Code Combination: Languages sharing the same written script (e.g. latin) can be used together.
      English can be used with any language.
    """

    def __init__(self, name="EasyOCR"):
        super().__init__()
        self.name = name
        # english and traditional chinese
        self.languages = ["en", "ch_tra"]
        self.load_model()

    def load_model(self):
        self.reader = easyocr.Reader(self.languages)

    def generate(self, image_data):
        text = ""
        try:
            text = " ".join(self.reader.readtext(image_data, detail=0))
            if text:
                text = "Text extracted from image:\n" + text
        except Exception as e:
            logger.error(f"Could not run EasyOCR due to the following error:\n{e}")

        return text


class LlavaNext(BaseModel):
    # for image to text on gpu
    # link: https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#single-image-inference
    # name="llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(self, name="llava-hf/llava-v1.6-mistral-7b-hf"):
        super().__init__()
        self.name = name
        self.default_prompt = textwrap.dedent(
            """
        Describe this image in detail.\nInclude any specific details on 
        background colors, patterns, themes, settings/context (for example if it's a search page 
        result, texting platform screenshot, pic of scenery etc.,), what might be going on in 
        the picture (activities, conversations), what all and how many objects, animals and people 
        are present, their orientations and activities, etc.,\n
        Besides a general description, include any details that might help uniquely identify the image."""
        )
        self.load_model()

    def load_model(self):
        self.processor = LlavaNextProcessor.from_pretrained(self.name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=512):
        if text is None:
            text = self.default_prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]
        image = image_data
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_length)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        return generated_text


# workaround for unnecessary flash_attn requirement
from unittest.mock import patch

from transformers.dynamic_module_utils import get_imports


def fixed_get_imports(filename: str) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


class Florence(BaseModel):
    # for image to text on gpu
    # link: https://huggingface.co/microsoft/Florence-2-large-ft
    # name="microsoft/Florence-2-large-ft"

    def __init__(self, name="microsoft/Florence-2-large-ft"):
        super().__init__()
        self.name = name
        # self.default_prompt = "Describe this image in detail."
        self.tasks = [
            "<OD>",
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<CAPTION_TO_PHRASE_GROUNDING>",
            "<DENSE_REGION_CAPTION>",
            "<REGION_PROPOSAL>",
            "<OCR>",
            "<OCR_WITH_REGION>",
        ]
        self.load_model()

    def load_model(self):
        if self.device == "cpu":
            # workaround for unnecessary flash_attn requirement on CPU
            with patch(
                "transformers.dynamic_module_utils.get_imports", fixed_get_imports
            ):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.name, torch_dtype=self.torch_dtype, trust_remote_code=True
                ).to(self.device)
        else:
            # on GPU, needs flash_attn. Install if not already done.
            import pip

            package = "flash_attn"
            try:
                __import__(package)
            except ImportError:
                logger.info(
                    f"Module {package} needed for Florence image model on GPU. Trying to `pip install {package}`."
                )
                pip.main(["install", package])
            # set model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.name, torch_dtype=self.torch_dtype, trust_remote_code=True
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.name, trust_remote_code=True
        )

    def generate(
        self,
        image_data,
        task_prompt="<MORE_DETAILED_CAPTION>",
        text=None,
        max_length=1024,
    ):
        # With additional text (for all tasks except <CAPTION_TO_PHRASE_GROUNDING>), gives error:
        # AssertionError: Task token <MORE_DETAILED_CAPTION> should be the only token in the text.

        if text is not None and task_prompt == "<CAPTION_TO_PHRASE_GROUNDING>":
            prompt = task_prompt + text
        else:
            prompt = task_prompt

        image = image_data

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_length,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return parsed_answer[task_prompt]


config = {
    "cpu": [
        {"model": Blip, "name": "Salesforce/blip-image-captioning-large"},
        {"model": Blip2, "name": "Salesforce/blip2-opt-2.7b"},
    ],
    "cuda": [
        {"model": Florence, "name": "microsoft/Florence-2-large-ft"},
        {"model": LlavaNext, "name": "llava-hf/llava-v1.6-mistral-7b-hf"},
        {"model": Blip2, "name": "Salesforce/blip2-opt-2.7b"},
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
class ImageToText(BaseModel):
    def __new__(cls, module="ImageToText"):
        m = try_load_models(module)
        # ImageModel(name)
        self = m["model"](name=m["name"])
        return self
