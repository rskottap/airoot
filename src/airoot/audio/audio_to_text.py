__all__ = [
    "AudioToText",
    "Whisper",
]

import json
import logging
import subprocess

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from airoot.base_model import BaseModel, get_default_model, set_default_model

logger = logging.getLogger("airoot.AudioToText")


class Whisper(BaseModel):
    # for speech to text
    # name="openai/whisper-large-v3"

    def __init__(self, name="openai/whisper-large-v3"):
        super().__init__()
        self.name = name
        self.load_model()

    def load_model(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.name)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def generate(self, sample):
        # sample: Audio file or array
        result = self.pipe(sample)
        return result["text"]


config = {
    "cpu": [{"model": Whisper, "name": "openai/whisper-large-v3"}],
    "cuda": [{"model": Whisper, "name": "openai/whisper-large-v3"}],
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
    def __new__(cls, type, module="AudioToText"):
        m = try_load_models(type, module)
        # AudioModel(name)
        self = m["model"](name=m["name"])
        return self
