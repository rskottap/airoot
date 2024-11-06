#!/usr/bin/env python3

import argparse
import io
import logging
import sys

import assure
from PIL import Image

from airoot.image import ImageToText

logger = logging.getLogger("airoot.ImageToText")


def parse_args():
    parser = argparse.ArgumentParser(description="ImageToText")
    parser.add_argument(
        "image_file_path",
        nargs="?",
        type=str,
        help="Path to image file",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="User prompt (like question) to process image with. If not provided, by default generates a description of the image. ONLY RECOMMENDED ON GPU.",
    )
    parser.add_argument(
        "-l",
        "--max-length",
        type=int,
        help="Max new tokens to generate. Defaults to 512 or 1024 based on model.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file to write text to. If not given, prints to stdout.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Detect if input is piped or image file is directly provided.
    if not sys.stdin.isatty():
        image_bytes = io.BytesIO(sys.stdin.buffer.read())
        image = Image.open(image_bytes).convert("RGB")
    else:
        # Raise error if no audio input is provided
        if not args.image_file_path:
            raise Exception(
                "Error: No image input provided. Provide image_file_path as a positional argument (or pipe image bytes into the command)."
            )
        image = Image.open(args.image_file_path.strip()).convert("RGB")

    model = ImageToText()
    if not args.max_length:
        text = model.generate(image, text=args.prompt) + "\n"
    else:
        text = (
            model.generate(image, text=args.prompt, max_length=args.max_length) + "\n"
        )

    # Write to output file if provided
    if args.output:
        with open(args.output.strip(), "w") as f:
            f.write(text)

    # If stdout is not connected to a terminal (i.e., there's a pipe), pipe the text
    if not sys.stdout.isatty():
        sys.stdout.buffer.write(text.encode())
    else:
        # no pipe and output file not provided, then print to stdout
        if not args.output:
            # logger.info("Output text is:")
            print(text)


if __name__ == "__main__":
    main()
