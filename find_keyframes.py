#!/usr/bin/env python

import argparse
import glob
import os
import cv2
import multiprocessing
import sys
import json
from dotenv import load_dotenv

load_dotenv()

template_path = os.getenv('TEMPLATE_PATH', './files/templates/')
process_count = int(os.getenv('PROCESS_COUNT', 4))
similarity_ratio = float(os.getenv('SIMILARITY_RATIO', 0.85))
kickback_ratio = float(os.getenv('KICKBACK_RATIO', 0.1))
skip_frames = int(os.getenv('SKIP_FRAMES_COUNT', 30))

template_in_file_pattern = os.getenv('TEMPLATE_IN_FILE_PATTERN', 'in*')
template_out_file_pattern = os.getenv('TEMPLATE_OUT_FILE_PATTERN', 'out*')

parser = argparse.ArgumentParser(
    prog='python find_keyframes.py',
    description='Find keyframes into a video based on template images.'
)
parser.add_argument('filename', help="Path to the video file to edit", type=str)
parser.add_argument('--silent', action=argparse.BooleanOptionalAction)


def main():
    args = parser.parse_args()
    silent = args.silent

    if not silent:
        print('find_keyframes 1.0 by wushaolin')

    path = os.path.abspath(args.filename)
    if not silent:
        print(' file to process: ' + path)

    templates_in = find_templates_files(template_in_file_pattern)
    templates_out = find_templates_files(template_out_file_pattern)

    cv2.setUseOptimized(True)

    in_grays = get_grays(templates_in)
    out_grays = get_grays(templates_out)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    split_total_frames = round(total_frames / process_count)
    split_frames_group = split_frames_for_threads(total_frames, split_total_frames)

    manager = multiprocessing.Manager()
    timecodes = manager.dict()

    processes = []
    for i in range(process_count):
        t = multiprocessing.Process(
            target=process_segment,
            args=(i, timecodes, split_frames_group[i], path, in_grays, out_grays)
        )
        processes.append(t)
        t.start()

    for t in processes:
        t.join()

    cv2.destroyAllWindows()
    keyframes = []
    for proc in timecodes.values():
        for frame in proc:
            keyframes.append(frame)

    sys.stdout.write(json.dumps({
        "file": path,
        "end_time": total_frames / fps,
        "keyframes": keyframes
    }))


def find_templates_files(pattern):
    files = glob.glob(f'{template_path}/{pattern}')

    out = []
    for file in files:
        out.append(os.path.basename(file))

    return out


def process_segment(thread_id, timecodes, segment, path, in_templates, out_templates):
    tc = []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    kickback = [None, None, 1]

    i = segment[0]
    last_frame_ok = False
    while cap.isOpened():

        if i > segment[1]:
            i = segment[1] - 1
            last_frame_ok = True
            if last_frame_ok:
                break

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        ret, frame = cap.read()
        cframe = cap.get(cv2.CAP_PROP_POS_FRAMES)  # retrieves the current frame number
        # print('Reading : ' + str(cframe) + ' in thread id : ' + str(thread_id))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        templates = [
            ['in', in_templates],
            ['out', out_templates],
        ]

        kickback = process_templates(tc, gray_frame, templates, kickback, cframe, fps)

        i = i + skip_frames

    cap.release()
    timecodes[i] = tc


def process_templates(timecodes, frame, g_templates, kickback, cframe, fps):
    if kickback[0] is None or kickback[1] is None:
        find = False
        i = 0
        for g_template in g_templates:
            j = 0
            for template in g_template[1]:
                max_val = get_similarity_ratio(frame, template)
                if max_val > similarity_ratio:
                    timecodes.append({
                        "frame": cframe,
                        "type": g_template[0],
                        "time": cframe / fps
                    })
                    kickback[0] = i
                    kickback[1] = j
                    kickback[2] = max_val - kickback_ratio
                    find = True
                    break
                j += 1
            if find:
                break
            i += 1
    else:

        max_val = get_similarity_ratio(frame, g_templates[kickback[0]][1][kickback[1]])
        # skip next frame until we get under the kickback value again
        if max_val < kickback[2]:
            kickback[0] = None
            kickback[1] = None
            kickback[2] = 1

    return kickback


def split_frames_for_threads(total_frames, split_total_frames):
    index = 0
    remaining_frames = total_frames
    out = []
    while remaining_frames > 0:
        start = index * split_total_frames
        end = (index + 1) * split_total_frames

        if abs(remaining_frames - split_total_frames) < split_total_frames:
            remaining_frames = 0
            end = total_frames

        out.append([
            start, end
        ])

        remaining_frames -= split_total_frames
        index += 1

    return out


def get_grays(files):
    out = []
    for file in files:
        search_for = cv2.imread(template_path + file, cv2.IMREAD_UNCHANGED)
        search_for_gray = cv2.cvtColor(search_for, cv2.COLOR_BGR2GRAY)
        out.append(search_for_gray)
    return out


def get_similarity_ratio(frame, template):
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val


if __name__ == '__main__':
    main()
