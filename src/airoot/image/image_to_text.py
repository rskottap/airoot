__all__ = [
    "ImageToText",
    "BlipCaption",
    "BlipVQA",
    "Blip2",
    "EasyOCR",
    "Llava",
    "LlavaNext",
    "Florence",
]

import json
import logging
import re
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
    BlipForQuestionAnswering,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.ImageToText")


class BlipCaption(BaseModel):
    # for image to text on cpu
    # link: https://huggingface.co/Salesforce/blip-image-captioning-large
    # https://huggingface.co/docs/transformers/en/model_doc/blip#transformers.BlipForConditionalGeneration

    # name="Salesforce/blip-image-captioning-large"
    def __init__(self, name="Salesforce/blip-image-captioning-large"):
        super().__init__()
        self.name = name
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=512):
        # NOTE: blip-image-captioning-large didn't do well with question like prompts, only conditional
        image = image_data
        inputs = self.processor(image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        generated_text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return generated_text


class BlipVQA(BaseModel):
    # for vqa on cpu
    # link: https://huggingface.co/Salesforce/blip-vqa-base
    # https://huggingface.co/docs/transformers/en/model_doc/blip#transformers.BlipForQuestionAnswering

    # name="Salesforce/blip-vqa-base"
    def __init__(self, name="Salesforce/blip-vqa-base"):
        super().__init__()
        self.name = name
        self.default_prompt = "What is this image about? What is going on? What all objects/people/animals are there in this image, if any?"
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = BlipForQuestionAnswering.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=512):
        image = image_data
        if text is None:
            text = self.default_prompt
        inputs = self.processor(image, text=text, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        generated_text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return generated_text


class Blip2(BaseModel):
    # link: https://huggingface.co/Salesforce/blip2-opt-2.7b
    # https://huggingface.co/docs/transformers/en/model_doc/blip-2#transformers.Blip2ForConditionalGeneration
    # name="Salesforce/blip2-opt-2.7b" # For BLIP2

    def __init__(self, name="Salesforce/blip2-opt-2.7b"):
        super().__init__()
        self.name = name
        self.template = "Question: {} Answer:"
        self.default_prompt = "What is this image about? What is going on? What all objects/people/animals are there in this image, if any?"
        self.load_model()

    def load_model(self):
        self.processor = Blip2Processor.from_pretrained(self.name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=1024):
        if text is not None:
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
                text = "Text extracted from image:\n" + text.strip()
        except Exception as e:
            logger.error(f"Could not run EasyOCR due to the following error:\n{e}")

        return text


class Llava(BaseModel):
    # for image to text on gpu
    # link: https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#single-image-inference
    # name="llava-hf/llava-1.5-7b-hf"

    def __init__(self, name="llava-hf/llava-1.5-7b-hf"):
        super().__init__()
        self.name = name
        self.default_prompt = textwrap.dedent(
            """
        Describe this image in detail.\n"""
        )
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.name)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)

    def generate(self, image_data, text=None, max_length=1024):
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
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        output = self.model.generate(
            **inputs, max_new_tokens=max_length, do_sample=False
        )
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        m = re.search("ASSISTANT:", generated_text)
        return generated_text[m.end() :].strip()


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
        the picture (activities, conversations), what all and how many objects, what all animals or people etc., 
        are present if any, and so on."""
        )
        self.load_model()

    def load_model(self):
        self.processor = LlavaNextProcessor.from_pretrained(self.name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=1024):
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
        generated_text = self.processor.decode(
            output[0], skip_special_tokens=True
        ).strip()
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
                pip.main(["install", "--no-build-isolation", package])
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
        {"model": BlipCaption, "name": "Salesforce/blip-image-captioning-large"},
        {"model": Florence, "name": "microsoft/Florence-2-large-ft"},
    ],
    "cuda": [
        {"model": LlavaNext, "name": "llava-hf/llava-v1.6-mistral-7b-hf"},
        {"model": Llava, "name": "llava-hf/llava-1.5-7b-hf"},
        # {"model": Blip2, "name": "Salesforce/blip2-opt-2.7b"},
        {"model": Florence, "name": "microsoft/Florence-2-large-ft"},
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
