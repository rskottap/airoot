#!/usr/bin/env python3

__all__ = ['main']

import sys

def main(argv=None):

    import argparse
    from kern import pdf

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser('pdf')
    parser.add_argument('path', nargs='?')
    args = parser.parse_args(argv)

    if args.path is not None:
        file = open(args.path, 'rb')
    else:
        file = sys.stdin.buffer

    text = pdf.to_text(file)
    print(text)

if __name__ == '__main__':
    main(sys.argv[1:])
