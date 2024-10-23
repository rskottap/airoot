#!/usr/bin/env python3

import argparse
import sys

import soundfile as sf

from airoot.audio import TextToAudio


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"Length in seconds needs to be a positive integer (int n>0). Passed {value}\n"
        )
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(description="TextToAudio")
    parser.add_argument("text", type=str, help="Text prompt for audio generation")
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
        help="Length of the music to generate in seconds. Default 5 seconds for cpu and 20 for gpu.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Detect if input is piped or text is directly provided.
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        text = args.text.strip()
    output_file = args.output.strip()

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
    return audio_array


if __name__ == "__main__":
    audio_output = main()
    # Send the audio output to stdout for piping
    sys.stdout.buffer.write(audio_output.tobytes())
