"""
GPU highly recommended for good quality and inference times.

Bark:
Very very slow on cpu.
- HF:     https://huggingface.co/docs/transformers/main/en/model_doc/bark
- GitHub: https://github.com/suno-ai/bark
- Speaker Library for voice presets:
          https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c

### AudioCraft Models
All audiocraft models NEED a gpu to run.

MusicGen:
- HF:     https://huggingface.co/docs/transformers/main/en/model_doc/musicgen_melody#text-only-conditional-generation
- GitHub: https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md

AudioGen:
- Github: https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md

Magnet:
- GitHub: https://github.com/facebookresearch/audiocraft/blob/main/docs/MAGNET.md

"""

__all__ = [
    "TextToAudio",
    "Bark",
    "MusicGen",
    "StableAudio1",
    "get_models",
]

import json
import logging
import subprocess

import torch
from diffusers import StableAudioPipeline
from transformers import AutoProcessor, BarkModel, MusicgenForConditionalGeneration

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("text_to_audio")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class MusicGen(BaseModel):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.model.config.audio_encoder.sampling_rate

    def load_model(self):
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=torch.float16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.name)

    def generate(self, text, max_new_tokens=256):
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        audio_array = self.model.generate(
            **inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens
        )
        audio_array = audio_array[0, 0].numpy()
        return audio_array


class Bark(BaseModel):
    # name = "suno/bark" # "suno/bark-small"

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.model.generation_config.sample_rate

    def load_model(self):
        self.model = BarkModel.from_pretrained(self.name, torch_dtype=torch.float16).to(
            self.device
        )

        if self.device == "cuda":
            # if using CUDA device, offload the submodels from GPU to CPU when they’re idle
            self.model.enable_cpu_offload()

        self.model = (
            self.model.to_bettertransformer()
        )  # Better Transformer optimization
        self.processor = AutoProcessor.from_pretrained(self.name)

    def generate(self, text, voice_preset="v2/en_speaker_6"):
        inputs = self.processor(text, voice_preset=voice_preset).to(self.device)
        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array


class StableAudio1(BaseModel):
    # for text to music on cpu TODO: Actually test on cpu
    # name = "stabilityai/stable-audio-open-1.0"

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.pipe.vae.sampling_rate

    def load_model(self):
        self.pipe = StableAudioPipeline.from_pretrained(
            self.name, torch_dtype=torch.float16
        ).to(self.device)
        self.generator = torch.Generator(self.device).manual_seed(0)

    def generate(self, text, negative_prompt="Low quality.", audio_end_in_s=10.0):
        audio = self.pipe(
            text,
            negative_prompt=negative_prompt,
            audio_end_in_s=audio_end_in_s,
            generator=self.generator,
        ).audios

        audio_array = audio[0].T.float().cpu().numpy()
        return audio_array


class XTTS(BaseModel):
    # from TTS.api import TTS

    # !WARNING: NEEDS Python<3.12
    # for text to speech on cpu TODO: Actually test on cpu
    # name = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.model.config.audio_encoder.sampling_rate

    def load_model(self):
        self.tts = TTS(self.name).to(self.device)

    def generate(self, text, speaker_wav, out_language="en"):
        wav = self.tts.tts(text=text, speaker_wav=speaker_wav, language=out_language)
        return wav


# Defaults for speech and music generation in the order they should be tried,
# i.e., decreasing memory usage if loading bigger fails (due to memory costraints)

# Recommended 16GB GPU memory for bark and musicgen-melody
# bark-small and musicgen-small can be run on smaller GPU memories
# MusicGen models need local GPU. Even bark-small on cpu is too slow.

config = {
    "cpu": {
        "speech": [{"model": "", "name": ""}],
        "music": [{"model": StableAudio1, "name": "stabilityai/stable-audio-open-1.0"}],
    },
    "cuda": {
        "speech": [
            {"model": Bark, "name": "suno/bark"},
            {"model": Bark, "name": "suno/bark-small"},
        ],
        "music": [
            {"model": MusicGen, "name": "facebook/musicgen-melody"},
            {"model": MusicGen, "name": "facebook/musicgen-small"},
        ],
    },
}


def get_models():
    return config


def try_load_models(type, module) -> dict:
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        defaults = config[device][type]
        for model in defaults:
            try:
                subprocess.run(
                    [
                        "python3",
                        "../test_load_model.py",
                        json.dumps(model),
                        "Hello, I am speaking.",
                    ],
                    capture_output=True,
                    check=True,
                )
                set_default_model(model, module)
                return model
            except Exception as e:
                logger.error(
                    f"Unable to load model {model['name']} on {device.capitalize()} due to error {e}\nTrying next model.",
                    exc_info=True,
                )
                continue

    # either device is cpu or cuda models failed to load
    defaults = config["cpu"][type]
    for model in defaults:
        try:
            subprocess.run(
                [
                    "python3",
                    "../test_load_model.py",
                    json.dumps(model),
                    "Hello, I am speaking.",
                ],
                capture_output=True,
                check=True,
            )
            set_default_model(model, module)
            return model
        except Exception as e:
            logger.error(
                f"Unable to load model {model['name']} on {device.capitalize()} due to error {e}\nTrying next model.",
                exc_info=True,
            )
            continue

    raise Exception(
        f"Unable to load any of the default models for {type}:\n \{json.dumps(defaults, indent=4)}"
    )


# Default
class TextToAudio(BaseModel):
    def __new__(cls, type, module="audio"):
        m = try_load_models(type, module)
        # AudioModel(name)
        self = m["model"](name=m["name"])
        return self
