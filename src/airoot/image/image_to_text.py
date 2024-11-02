__all__ = [
    "ImageToText",
    "Blip",
    "InstructBlip",
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
    BlipForConditionalGeneration,
    BlipProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.ImageToText")


class Blip(BaseModel):
    # for image to text on cpu
    # link: https://huggingface.co/Salesforce/blip-image-captioning-large
    # name="Salesforce/blip-image-captioning-large"

    def __init__(self, name="Salesforce/blip-image-captioning-large"):
        super().__init__()
        self.name = name
        self.default_prompt = "Describe this image in detail."
        self.load_model()

    def load_model(self):
        self.processor = BlipProcessor.from_pretrained(self.name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_path, text=None, max_length=512):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.processor.decode(out[0], skip_special_tokens=True)
        return generated_text


class InstructBlip(BaseModel):
    # for image to text on big gpu, EXTRA
    # link: https://huggingface.co/Salesforce/instructblip-vicuna-7b
    # NOTE: 32 GB model !!!
    # name="Salesforce/instructblip-vicuna-7b"

    def __init__(self, name="Salesforce/instructblip-vicuna-7b"):
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
        self.processor = InstructBlipProcessor.from_pretrained(self.name)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_path, text=None, max_length=512):
        if text is None:
            text = self.default_prompt
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=max_length,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].strip()
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

    def generate(self, image_path):
        text = ""
        try:
            text = " ".join(self.reader.readtext(image_path, detail=0))
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

    def generate(self, image_path, text=None, max_length=512):
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
        image = Image.open(image_path).convert("RGB")
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
        image_path,
        task_prompt="<DETAILED_CAPTION>",
        text_input=None,
        max_length=1024,
    ):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        image = Image.open(image_path).convert("RGB")

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
        return parsed_answer


class Pixtral(BaseModel):
    # for image to text on gpu, EXTRA
    # link: https://huggingface.co/mistralai/Pixtral-12B-2409
    # pip install vllm mistral_common
    # NOTE: 25GB model
    # name="mistralai/Pixtral-12B-2409"

    def __init__(self, name="mistralai/Pixtral-12B-2409"):
        from vllm import LLM
        from vllm.sampling_params import SamplingParams

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
        self.llm = LLM(model=self.name, tokenizer_mode="mistral")

    def generate(self, image_url, text=None, max_length=1024):
        if text is None:
            text = self.default_prompt
        # image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        self.sampling_params = SamplingParams(max_tokens=max_length)
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text


config = {
    "cpu": [
        {"model": Blip, "name": "Salesforce/blip-image-captioning-large"},
    ],
    "cuda": [
        {"model": Florence, "name": "microsoft/Florence-2-large-ft"},
        {"model": LlavaNext, "name": "llava-hf/llava-v1.6-mistral-7b-hf"},
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
