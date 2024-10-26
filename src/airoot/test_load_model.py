#!/usr/bin/env python3
"""
Tries to load the model into memory (maybe run a sample generation).
The internals of model init handle which device to fit to.
"""
import argparse

import airoot
from airoot.base_model import get_models


def main():
    parser = argparse.ArgumentParser(description="test_load_model")
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        required=True,
        help="The module of airoot to test. Should be one of the modules in airoot.etc.modules i.e., one of TextTo{x} or {x}ToText for x in Audio, Video, Image, Text.",
    )
    parser.add_argument(
        "-k",
        "--keys",
        type=str,
        nargs="*",
        required=True,
        help="The keys IN ORDER, to index into the modules 'model config' dictionary to get the list of default models. For ex, 'cpu' and 'speech' to test the audio models for cpu, for speech to text.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="The index of the model in the available list of default models for a given config.",
    )
    args = parser.parse_args()

    model_config = get_models(args.module)
    model = model_config
    for key in args.keys:
        model = model[key]
    model = model[args.idx]

    m = model["model"](name=model["name"])
    # _ = m.generate(args.text)


if __name__ == "__main__":
    main()
