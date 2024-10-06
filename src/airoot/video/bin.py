#!/usr/bin/env python3

__all__ = ['main']

import sys

def main(argv=None):

    import argparse
    import kern

    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser('video')
    parser.add_argument('query', nargs='?')
    args = parser.parse_args(argv)

    file = sys.stdin.buffer
    bytes = file.read()

    if args.query is None:
        text = kern.video_to_text(bytes)
    else:
        text = kern.video_and_text_to_text(bytes, args.query)
    print(text)

if __name__ == '__main__':
    main(sys.argv[1:])
