### image-to-text, vqa


import textwrap

import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

from airoot.base_model import BaseModel


class DeepSeekJanus(BaseModel):
    # HF link: https://huggingface.co/deepseek-ai/Janus-Pro-7B
    # GitHub link: https://github.com/deepseek-ai/Janus
    # name="deepseek-ai/Janus-Pro-7B"

    def __init__(self, name="deepseek-ai/Janus-Pro-1B"):
        super().__init__()
        self.name = name
        self.default_prompt = textwrap.dedent(
            """
        Describe this image in detail.\n"""
        )
        self.load_model()

    def load_model(self):
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.name
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.name, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cpu().eval()

    def generate(self, image_data, text=None, max_length=1024):
        if text is None:
            text = self.default_prompt
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{text}",
                "images": [image_data],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_length,
            do_sample=False,
            use_cache=True,
        )

        generated_text = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        return generated_text
