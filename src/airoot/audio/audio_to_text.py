__all__ = [
    "AudioToText",
    "Whisper",
    "Whisper2",
]

import json
import logging
import subprocess

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.AudioToText")


class Whisper(BaseModel):
    # for speech to text
    # The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration.
    # name="openai/whisper-large-v2" # name="openai/whisper-medium" # name="openai/whisper-small"

    def __init__(self, name="openai/whisper-small"):
        super().__init__()
        self.name = name
        self.load_model()

    def load_model(self):
        self.processor = WhisperProcessor.from_pretrained(self.name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.name)
        self.model.config.forced_decoder_ids = None
        self.model.to(self.device)

    def generate(self, data, sample_rate, language=None, task="transcribe"):
        # If language or task is not provided, automatically detects source language and translates to English
        if language is not None:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task=task
            )
        else:
            forced_decoder_ids = None

        input_features = self.processor(
            data, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features
        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids
        )
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text


class Whisper2(BaseModel):
    # for speech to text
    # For samples longer than 30 seconds use the chunking algorithm via the "automatic-speech-recognition" pipeline
    # name="openai/whisper-large-v2" # name="openai/whisper-medium" # name="openai/whisper-small"

    def __init__(self, name="openai/whisper-small"):
        super().__init__()
        self.name = name
        self.load_model()

    def load_model(self):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.name,
            chunk_length_s=30,
            device=self.device,
        )
        self.processor = WhisperProcessor.from_pretrained(self.name)
        self.original_decoder_ids = self.pipe.model.generation_config.forced_decoder_ids

    def generate(self, data, task="transcribe"):
        # Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English.
        # If you want to instead always translate your audio to English, make sure to pass `language='en'`
        if task == "translate":
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="english", task=task
            )
            self.pipe.model.generation_config.forced_decoder_ids = forced_decoder_ids
            text = self.pipe(data)["text"]
            # reset back for next generations
            self.pipe.model.generation_config.forced_decoder_ids = (
                self.original_decoder_ids
            )
        else:
            # by default detects language and transcribes into that language
            text = self.pipe(data)["text"]
        return text


config = {
    "cpu": [
        {"model": Whisper2, "name": "openai/whisper-medium"},
        {"model": Whisper2, "name": "openai/whisper-small"},
    ],
    "cuda": [{"model": Whisper2, "name": "openai/whisper-large-v2"}],
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
class AudioToText(BaseModel):
    def __new__(cls, module="AudioToText"):
        m = try_load_models(module)
        # AudioModel(name)
        self = m["model"](name=m["name"])
        return self
