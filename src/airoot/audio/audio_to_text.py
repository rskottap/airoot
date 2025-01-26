__all__ = [
    "AudioToText",
    "Whisper",
    "Whisper2",
]

import logging

from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from airoot.base_model import BaseModel, try_load_models

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


# Default
class AudioToText(BaseModel):
    def __new__(cls, module="AudioToText"):
        m = try_load_models(module)
        # AudioModel(name)
        self = m["model"](name=m["name"])
        return self
