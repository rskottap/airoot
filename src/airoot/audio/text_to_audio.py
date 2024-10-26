"""
GPU highly recommended for good quality and inference times.
See TextToAudio.md for model links and other details.
"""

__all__ = [
    "TextToAudio",
    "Bark",
    "MusicGen",
    "StableAudio1",
    "ParlerTTS",
]

import json
import logging
import subprocess

import torch
from diffusers import StableAudioPipeline
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BarkModel,
    MusicgenForConditionalGeneration,
)

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.TextToAudio")


class MusicGen(BaseModel):
    # for text to music on cpu/gpu
    # name="facebook/musicgen-small" # name="facebook/musicgen-melody"

    def __init__(self, name="facebook/musicgen-small"):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.model.config.audio_encoder.sampling_rate
        self.frame_rate = self.model.config.audio_encoder.frame_rate

    def load_model(self):
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.name,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.name)

    def generate(self, text, audio_end_in_s=5.0):
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        audio_array = self.model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3,
            max_new_tokens=int(audio_end_in_s * self.frame_rate),
        )
        audio_array = audio_array[0, 0].cpu().numpy()
        return audio_array


class Bark(BaseModel):
    # for text to speech/music on gpu
    # name = "suno/bark" # name = "suno/bark-small"

    def __init__(self, name="suno/bark-small"):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.model.generation_config.sample_rate

    def load_model(self):
        self.model = BarkModel.from_pretrained(
            self.name,
        ).to(self.device)

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
    # for text to music on gpu
    # name = "stabilityai/stable-audio-open-1.0"

    def __init__(self, name="stabilityai/stable-audio-open-1.0"):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.pipe.vae.sampling_rate

    def load_model(self):
        self.pipe = StableAudioPipeline.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)
        self.generator = torch.Generator(self.device).manual_seed(0)

    def generate(self, text, audio_end_in_s=10.0, negative_prompt="Low quality."):
        audio = self.pipe(
            text,
            negative_prompt=negative_prompt,
            audio_end_in_s=audio_end_in_s,
            generator=self.generator,
        ).audios

        audio_array = audio[0].T.float().cpu().numpy()
        return audio_array


class ParlerTTS(BaseModel):
    # for text to speech on cpu
    # name = "parler-tts/parler-tts-mini-v1"

    def __init__(self, name="parler-tts/parler-tts-mini-v1"):
        super().__init__()
        self.name = name
        # A default description
        self.description = "{person}'s voice, with a very close recording that almost has no background noise and very clear audio."
        self.load_model()
        self.sample_rate = self.model.config.sampling_rate

    def load_model(self):
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.name, torch_dtype=self.torch_dtype
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    def generate(self, text, voice_preset="Jon", description=None):
        if description is None:
            description = self.description.format(person=voice_preset)

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(
            self.device
        )
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )
        generation = self.model.generate(
            input_ids=input_ids, prompt_input_ids=prompt_input_ids
        )
        audio_array = generation.cpu().numpy().squeeze()
        return audio_array


# Defaults for speech and music generation in the order they should be tried,
# i.e., decreasing memory usage if loading bigger fails (due to memory costraints)

# Recommended 16GB GPU memory for bark and musicgen-melody
# bark-small and musicgen-small can be run on smaller GPU memories
# MusicGen models need local GPU. Even bark-small on cpu is too slow.

config = {
    "cpu": {
        "speech": [{"model": ParlerTTS, "name": "parler-tts/parler-tts-mini-v1"}],
        "music": [{"model": MusicGen, "name": "facebook/musicgen-small"}],
    },
    "cuda": {
        "speech": [
            {"model": Bark, "name": "suno/bark"},
            {"model": Bark, "name": "suno/bark-small"},
        ],
        "music": [
            {"model": StableAudio1, "name": "stabilityai/stable-audio-open-1.0"},
            {"model": MusicGen, "name": "facebook/musicgen-medium"},
        ],
    },
}


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
    model = get_default_model(module, type)
    if model:
        return model
    logger.info(
        f"No default model found (in ~/.cache/airoot) for {module}/{type} on this device. Trying to determine default model for this device."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # order in which to try out the models
    params = []

    # first try to load cuda models
    if device == "cuda":
        defaults = config["cuda"][type]
        params.extend(
            ["--keys", "cuda", type, "--idx", i] for i in range(len(defaults))
        )
    # either device is cpu or cuda models failed to load
    defaults = config["cpu"][type]
    params.extend(["--keys", "cpu", type, "--idx", i] for i in range(len(defaults)))

    import os

    import airoot

    for p in params:
        try:
            model = config[p[1]][p[2]][p[4]]
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
            set_default_model([p[1], p[2], p[4]], module, model, type)
            return model
        except Exception as e:
            logger.error(
                f"Unable to load model {model['name']} on {device.upper()} due to error:\n{e.stderr}\nTrying next model.",
                exc_info=True,
            )
            continue

    raise Exception(
        f"Unable to load any of the default models in {module} module for {type}. All available models:\n{json.dumps(config, default=lambda o: str(o), indent=2)}"
    )


# Default
class TextToAudio(BaseModel):
    def __new__(cls, type, module="TextToAudio"):
        m = try_load_models(type, module)
        # AudioModel(name)
        self = m["model"](name=m["name"])
        return self
