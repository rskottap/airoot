#!/usr/bin/env python3
"""
Tries to load the model into memory and run a sample generation.
The internals of model init handle which device to fit to.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="test_load_model")
    parser.add_argument(
        "-m",
        "--module",
        type=str,
        required=True,
        help="The module of airoot to test. One of audio, video, image and text.",
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
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=True,
        help="Text prompt for sample generation.",
    )
    args = parser.parse_args()

    if args.module == "audio":
        from airoot import audio

        model_config = audio.get_models()
        model = model_config
        for key in args.keys:
            model = model[key]
        model = model[args.idx]

        m = model["model"](name=model["name"])
        # _ = m.generate(args.text)


if __name__ == "__main__":
    main()
