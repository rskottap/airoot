__all__ = [
    "BaseModel",
    "set_default_model",
    "get_default_model",
    "get_model_config",
    "try_load_models",
]

import json
import logging
import os
import subprocess
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
        f"Set model {model} with keys {model_keys} as the default model for {os.path.join(module, *args)} module in {str(p)}"
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
        config = get_model_config(module)
        model = config
        for key in model_keys:
            model = model[key]

        logger.info(f"Found default model {model} in {p}")
        return model
    return None


def get_model_config(module: str) -> dict:
    if module == "TextToAudio":
        from airoot.audio.text_to_audio import config

        return config
    if module == "AudioToText":
        from airoot.audio.audio_to_text import config

        return config
    if module == "ImageToText":
        from airoot.image.image_to_text import config

        return config
    if module == "TextToImage":
        from airoot.image.text_to_image import config

        return config


def try_load_models(module, *extra_keys) -> dict:
    """
        extra_keys: Any additional keys (like speech/music) *in order* besides "cpu" and "cuda" to be able to index into the modules 'config' dict to get the end list of default models List(Dict()): [{"model": <class>, "name": <str: HF model path>}, {},...]

    Tries to load the model into memory, in order of devices.

    Does this by trying to load the model into memory in a separate process. So if it fails mid-way, with some model layers loaded into memory but not all, and raises an exception, the GPU memory gets cleared up automatically when the process exits. Otherwise, we won't have access to the model class/variable to be able to delete it later and clear up memory. Hence, trying it in a different process.

    If successful, writes that model to etc.cache_path as the default model to use on that machine for the module [audio, video, image, text].

    If gpu is available then:
        - uses gpu regardless of model.
        - first try to load the cuda models in order (decreasing memory usage).
        - If all the cuda models fail, then switch to cpu default models but fit them to gpu.

    If no gpu is available then try to load the cpu models in order.
    """
    extra_keys = [str(k) for k in extra_keys]

    model = get_default_model(module, *extra_keys)
    if model:
        return model
    logger.info(
        f"No default model found (in ~/.cache/airoot) for {os.path.join(module, *extra_keys)} on this device. Trying to determine default model for this device."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(module)

    # order in which to try out the models
    # first try to load cuda models
    if device == "cuda":
        defaults = config["cuda"]
    else:
        # either device is cpu or cuda models failed to load
        defaults = config["cpu"]

    for key in extra_keys:
        defaults = defaults[key]

    import airoot

    test_path = os.path.join(airoot.__path__[-1], "test_load_model.py")
    base_command = [
        "python3",
        test_path,
        "-m",
        module,
    ]

    for i in range(len(defaults)):
        model = defaults[i]
        try:
            # parameters to test_load_model script
            params = ["--keys", device, *extra_keys, "--idx", i]
            params = [str(x) for x in params]
            command = base_command + params
            _ = subprocess.run(
                command,
                capture_output=True,
                check=True,
            )
            set_default_model([device, *extra_keys, i], module, model, *extra_keys)
            return model
        except Exception as e:
            logger.error(
                f"Unable to load model {model['name']} on {device.upper()} due to error:\n{e.stderr}\nClearing HF cache and Trying next model.",
                exc_info=True,
            )
            # TODO: Clear HF Cache for failed to load model but was downloaded successfully
            continue

    raise Exception(
        f"Unable to load any of the default models for {os.path.join(module, *extra_keys)} module. All available models:\n{json.dumps(config, default=lambda o: str(o), indent=2)}"
    )
