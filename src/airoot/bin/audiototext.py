#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import librosa

from airoot.audio import AudioToText

logger = logging.getLogger("airoot.AudioToText")


def parse_args():
    parser = argparse.ArgumentParser(description="AudioToText")
    parser.add_argument(
        "audio_file_path",
        nargs="?",
        type=str,
        help="Path to audio file to transcribe/translate (.wav files recommended)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Language of source audio. If not given, auto-infers source audio and TRANSLATES from source audio language to English. If given, by DEFAULT, TRANSCRIBES to text in same language as audio file.",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform: 'transcribe' or 'translate' (default: 'transcribe'). If language is specified, for transcription, converts audio in language x to text in x. For translation converts audio in language x to text in english.",
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
        audio = sys.stdin.read()
        audio, sr = librosa.load(audio, sr=target_sr)
    else:
        if not args.audio_file_path.endswith("wav"):
            logger.warning(
                f"Recommended to use .wav audio files for input. Gave {args.audio_file_path}"
            )

        audio, sr = librosa.load(args.audio_file_path, sr=target_sr)

    # Raise error if no audio input is provided
    if not audio:
        raise Exception(
            "Error: No audio input provided. Provide audio_file_path as a positional argument or pipe an audio bytes into the command."
        )
    model = AudioToText()
    text = model.generate(audio, sr, language=args.language, task=args.task)

    # Write to output file if provided
    if args.output:
        with open(args.output.strip(), "w") as f:
            f.write(text)

    # If stdout is not connected to a terminal (i.e., there's a pipe), output the text as raw bytes
    if not sys.stdout.isatty():
        sys.stdout.buffer.write(text.tobytes())
    else:
        # no pipe and output file not provided, then print to stdout
        if not args.output:
            print(text)


if __name__ == "__main__":
    main()
