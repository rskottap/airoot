import textwrap

from transformers import (
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)

from airoot.base_model import BaseModel


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

    def generate(self, image_data, text=None, max_length=512):
        if text is None:
            text = self.default_prompt
        image = image_data

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


class Vilt(BaseModel):
    # for image to text， very basic VQA on cpu
    # link: https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
    # name="dandelin/vilt-b32-finetuned-vqa"

    def __init__(self, name="dandelin/vilt-b32-finetuned-vqa"):
        super().__init__()
        self.name = name
        self.default_prompt = "Describe this image in detail."
        self.load_model()

    def load_model(self):
        self.processor = ViltProcessor.from_pretrained(self.name)
        self.model = ViltForQuestionAnswering.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)

    def generate(self, image_data, text=None, max_length=512):
        if text is None:
            text = self.default_prompt
        image = image_data

        inputs = self.processor(image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        generated_text = self.model.config.id2label[idx]
        return generated_text


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
        # image = image_data
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
