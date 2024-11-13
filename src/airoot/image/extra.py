import textwrap

from transformers import (
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)

from airoot.base_model import BaseModel

################## ImageToText ##################


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


################## TextToImage ##################
from diffusers import DiffusionPipeline


class SDXLBase(BaseModel):
    # for text to image on gpu
    # HF docs: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline
    # Model card: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    # Pipelines: https://huggingface.co/docs/diffusers/using-diffusers/sdxl#stable-diffusion-xl

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
        init_image=None,
        mask_image=None,
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
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=n_steps,
            denoising_end=denoising_end,
            original_size=target_size,
            target_size=target_size,
            output_type=output_type,
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
        prompt,
        prompt_2=None,
        negative_prompt=None,
        init_image=None,
        mask_image=None,
        n_steps=40,
        denoising_end=0.8,
        target_size=(1024, 1024),
    ):
        # returns PIL.Image.Image type
        refined_image = self.refiner(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=n_steps,
            denoising_start=denoising_end,
            original_size=target_size,
            target_size=target_size,
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
        init_image=None,
        mask_image=None,
        n_steps=40,
        denoising_end=0.8,
        target_size=(1024, 1024),
    ):
        # returns PIL.Image.Image type
        if init_image is None:
            init_image = self.base.generate(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask_image=mask_image,
                n_steps=n_steps,
                denoising_end=denoising_end,
                output_type="latent",
                target_size=target_size,
            )

        refined_image = self.refiner.generate(
            init_image=init_image,
            mask_image=mask_image,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            n_steps=n_steps,
            denoising_end=denoising_end,
            target_size=target_size,
        )
        return refined_image
