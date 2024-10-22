__all__ = ["BaseModel", "set_default_model", "get_default_model"]

import json
import logging
import os
from pathlib import Path

import torch

from airoot.etc import cache_path

logger = logging.getLogger("airoot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        raise NotImplementedError(
            "load_model() must be implemented in the derived class"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "run_inference() must be implemented in the derived class"
        )


def set_default_model(model: dict, module: str):
    p = Path(os.path.join(cache_path, module)).expanduser()
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "model.json")) as f:
        f.write(json.dumps(model))
    logger.info(
        f"Set model {model} as the default model for {module} module in {str(p)}"
    )


def get_default_model(module: str) -> dict | None:
    p = Path(os.path.join(cache_path, module, "model.json")).expanduser()
    if p.exists():
        model = json.load(open(p))
        logger.info(f"Found default model {model} for {module} module in {p}")
        return model
    return None
