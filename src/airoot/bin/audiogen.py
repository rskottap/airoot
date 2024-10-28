#!/usr/bin/env python3

import argparse
import io
import logging
import sys

import soundfile as sf

from airoot.audio import TextToAudio

logger = logging.getLogger("airoot.TextToAudio")


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"Length in seconds needs to be a positive integer (int n>0). Passed {value}\n"
        )
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(description="TextToAudio")
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        help="Text prompt for audio generation. Can be piped in or directly passed.",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["speech", "music"],
        default="speech",
        help="Type of audio to generate: 'speech' or 'music' (default: 'speech')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Output file to write the audio stream to (default: output.wav)",
    )

    # Speech-specific option
    parser.add_argument(
        "-vp", "--voice-preset", type=str, help="Voice preset for speech generation"
    )
    # Music-specific option
    parser.add_argument(
        "-n",
        "--len",
        type=check_positive,
        help="Length of the music to generate in seconds. Default 5 seconds.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Detect if input is piped or text is directly provided.
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        text = args.text.strip()
    # Raise error if no text input is provided
    if not text:
        raise Exception(
            "Error: No text input provided. Provide text as a positional argument or pipe it into the command."
        )

    output_file = args.output.strip()
    if not output_file.endswith("wav"):
        new_output_file = output_file.split(".")[0] + ".wav"
        logger.warning(
            f"Output file to write to must be a .wav file but gave '{output_file}'. Writing to '{new_output_file}' instead."
        )
        output_file = new_output_file

    audio_array = []
    model = None
    if args.type == "music":
        model = TextToAudio("music")
    else:
        model = TextToAudio("speech")

    if args.type == "speech" and args.voice_preset and args.voice_preset.strip():
        audio_array = model.generate(text, voice_preset=args.voice_preset.strip())
    elif args.type == "music" and args.len:
        audio_array = model.generate(text, audio_end_in_s=args.len)
    else:
        audio_array = model.generate(text)

    sf.write(output_file, audio_array, model.sample_rate)

    # If stdout is not connected to a terminal (i.e., there's a pipe), write audio data as WAV format to stdout
    if not sys.stdout.isatty():
        # Write to an in-memory buffer first
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, model.sample_rate, format="WAV")

        # Move to the start of the buffer
        buffer.seek(0)

        # Write the buffer contents to stdout
        sys.stdout.buffer.write(buffer.read())
        sys.stdout.flush()  # Ensure all data is written


if __name__ == "__main__":
    main()
