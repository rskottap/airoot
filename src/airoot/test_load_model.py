"""
Tries to load the model into memory and run a sample generation.
The internals of model init handle which device to fit to.
"""
import argparse

from airoot.audio import *


def main():
    parser = argparse.ArgumentParser(description="test_load_model")
    parser.add_argument(
        "model_config",
        type=dict,
        help="The model config to load {'model': <model class>, 'name': <model name>}.",
    )
    parser.add_argument("text", type=str, help="Text prompt for sample generation.")
    args = parser.parse_args()

    model = args.model_config
    m = model["model"](name=model["name"])
    _ = m.generate(args.text)
