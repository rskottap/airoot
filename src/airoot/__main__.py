#!/usr/bin/env python3

__all__ = ['main']

import os
import sys
import kern
import shutil
import argparse
import subprocess

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser('input')
    parser.add_argument('query', nargs='?')
    args = parser.parse_args(argv)

    query = (args.query,) if args.query else ()

    if args.query:
        query = (args.query,)
    else:
        query = ()

    blobs_with_types = []
    for stream in streams:
        blob = stream.read()

        type = None
        if kern.is_image(blob):
            type = 'image'
        if kern.is_video(blob):
            type = 'video'
        elif kern.is_pdf(blob):
            type = 'pdf'
        elif kern.is_doc(blob):
            type = 'doc'
        elif kern.is_html(blob):
            type = 'html'
        elif kern.is_url(blob):
            type = 'url'
        else:
            raise TypeError(f"Not sure how to convert to text: {blob!r}")
        proc = subprocess.Popen([f"input-{type}", *query], stdin=subprocess.PIPE)
        proc.communicate(blob)

if __name__ == '__main__':
    main(sys.argv[1:])
