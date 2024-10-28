#!/usr/bin/env python3

import argparse
import io
import logging
import sys

import librosa
import soundfile as sf

from airoot.audio import AudioToText

logger = logging.getLogger("airoot.AudioToText")


def parse_args():
    parser = argparse.ArgumentParser(description="AudioToText")
    parser.add_argument(
        "audio_file_path",
        nargs="?",
        type=str,
        help="Path to audio file (.wav files recommended!)",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: 'transcribe' or 'translate' (default: 'transcribe').\nAuto-detects language, and based on the task, either transcribes audio to text in that language or translates to English.",
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

    # Detect if input is piped or audio file is directly provided.
    # Load the input audio and Re-sample the audio to 16kHz for Whisper inference
    target_sr = 16000

    if not sys.stdin.isatty():
        audio = io.BytesIO(sys.stdin.buffer.read())
        # audio, _ = sf.read(sys.stdin.buffer)
        audio, _ = librosa.load(audio, sr=target_sr)
    else:
        # Raise error if no audio input is provided
        if not args.audio_file_path:
            raise Exception(
                "Error: No audio input provided. Provide audio_file_path as a positional argument (or pipe audio bytes into the command)."
            )
        if not args.audio_file_path.endswith("wav"):
            logger.warning(
                f"Recommended to use .wav audio files for input. Gave {args.audio_file_path}"
            )

        audio, _ = librosa.load(args.audio_file_path.strip(), sr=target_sr)

    model = AudioToText()
    text = model.generate(audio, task=args.task) + "\n"

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
            logger.info("Output text is:")
            print(text)


if __name__ == "__main__":
    main()
