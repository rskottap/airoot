__all__ = ["BaseModel", "set_default_model", "get_default_model", "get_models"]

import json
import logging
import os
from pathlib import Path

import colorlog
import torch

from airoot import etc

logger = logging.getLogger("airoot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(name)s - %(levelname)s - %(message)s", log_colors=etc.log_colors
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def load_model(self):
        raise NotImplementedError(
            "load_model() must be implemented in the derived class"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "run_inference() must be implemented in the derived class"
        )


def set_default_model(model_keys: list[str | int], module: str, model: dict, *args):
    """
    model_keys: The list of keys in order to be able to index into that modules *config* dict to get the required model.
                since the models are of type <class>, just using the keys to index into config avoids import errors and json dumps conversions to/from strings.
    module:     Name of module. Should be one of etc.modules
    model:      The model dict, only for better logs
    args:       Any other args get appended to the path `cache_path/module/....` for arbitrary nesting and types
                for example: speech and music categories in TextToAudio
    """

    args = [str(a) for a in args]
    p = Path(os.path.join(etc.cache_path, module, *args)).expanduser()
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "model.keys"), "w") as f:
        f.write(",".join(model_keys))
    logger.info(
        f"Set model {model} with keys {model_keys} as the default model for {module} module in {str(p)}"
    )


def eval_if_int(value):
    try:
        result = eval(value)
        if isinstance(result, int):
            return result
    except:
        return value
    return value


def get_default_model(module: str, *args) -> dict | None:
    args = [str(a) for a in args]
    p = Path(os.path.join(etc.cache_path, module, *args, "model.keys")).expanduser()
    if p.exists():
        model_keys = open(p).readline().strip().split(",")
        model_keys = [eval_if_int(k) for k in model_keys]  # eval any ints for indexing

        # index into config till you get to the model dict {model: <class>, name: "name"}
        config = get_models(module)
        model = config
        for key in model_keys:
            model = model[key]

        logger.info(f"Found default model {model} in {p}")
        return model
    return None


def get_models(module: str) -> dict:
    if module == "TextToAudio":
        from airoot.audio.text_to_audio import config

        return config
    if module == "AudioToText":
        from airoot.audio.audio_to_text import config

        return config
