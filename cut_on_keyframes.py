#!/usr/bin/env python

import argparse
import datetime
import json
import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

ffmpeg_path = os.getenv('FFMPEG_PATH', 'ffmpeg')

parser = argparse.ArgumentParser(
                    prog='python cut_on_keygrames.py',
                    description='Cut a video based on the keyframes received from stdin.'
)
parser.add_argument('--out', help='Directory where the video will be exported', default='./files/out')
parser.add_argument('--keyframes', help='Json object holding keyframes and file path', default=None)
parser.add_argument('--silent', action=argparse.BooleanOptionalAction)


def main():
    args = parser.parse_args()
    silent = args.silent

    if not silent:
        print('cut_on_keyframes 1.0 by wushaolin')

    out = args.out
    inputs = args.keyframes
    if inputs is None:
        inputs = read_from_stdin()

    inputs = json.loads(inputs)
    file = inputs["file"]
    eof = inputs["end_time"]

    outfile = create_unique_filename(os.path.join(out, os.path.basename(file)))
    # print(outfile)

    # print('File to process : ' + file)
    # print('Keyframes : ' + json.dumps(inputs["keyframes"]))

    keyframes = sorted(inputs["keyframes"], key=lambda d: d['frame'])

    # print('Sorted keyframes : ' + json.dumps(keyframes))

    loop = True
    while loop:
        in_keyframe = pop_first_of_type(keyframes, 'in')
        out_keyframe = pop_first_of_type(keyframes, 'out')

        start_at = int(in_keyframe['time'])
        if out_keyframe is not None:
            duration = int(out_keyframe['time']) - start_at
        else:
            duration = int(eof) - start_at

        command = [
            ffmpeg_path,
            '-ss', format_seconds(start_at),
            '-i', file,
            '-to', format_seconds(duration),
            '-c', 'copy',
            outfile
        ]

        subprocess.run(command)

        if not keyframes:
            loop = False

    # print('end')


def read_from_stdin():
    # Read input from stdin
    for line in sys.stdin:
        # Process each line as needed
        return line.strip()


def format_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_unique_filename(filename):
    if os.path.exists(filename):
        name, ext = os.path.splitext(filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"
        return new_filename
    else:
        return filename


def pop_first_of_type(keyframes, entry_type):
    first_index = None
    for i, item in enumerate(keyframes):
        if item["type"] == entry_type:
            first_index = i
            break

    if first_index is not None:
        return keyframes.pop(first_index)
    else:
        return None


if __name__ == '__main__':
    main()
