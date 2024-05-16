import cv2
import multiprocessing

template_path = './files/templates/'
# video_path = './files/in/2024-05-13 22-36-24.mkv'
video_path = './files/in/2024-05-15_W_Blitz_2v2_SSvSS_Again.mp4'
thread_count = 4
similarity_ratio = 0.85
kickback_ratio = 0.1

templates_in = [
    'in_template_sov.png',
    'in_template_allied.png'
]
templates_out = [
    'out_template_sov.png',
    'out_template_allied.png'
]

timecodes = []


def main():
    print('RA2 - SMART VIDEO CUT 1.0')

    cv2.setUseOptimized(True)

    in_grays = get_grays(templates_in)
    out_grays = get_grays(templates_out)

    cap = cv2.VideoCapture(video_path)
    total_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    split_total_frames = round(total_frames / thread_count)
    split_frames_group = split_frames_for_threads(total_frames, split_total_frames)

    processes = []
    for i in range(thread_count):
        t = multiprocessing.Process(target=process_segment, args=(i, split_frames_group[i], video_path, in_grays, out_grays))
        t.start()
        processes.append(t)

    for t in processes:
        t.join()

    cv2.destroyAllWindows()
    print(timecodes)


def process_segment(thread_id, segment, path, in_templates, out_templates):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, segment[0])
    fps = cap.get(cv2.CAP_PROP_FPS)

    kickback = [None, None, 1]
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        cframe = cap.get(cv2.CAP_PROP_POS_FRAMES)  # retrieves the current frame number
        # print('Reading : ' + str(cframe) + ' in thread id : ' + str(thread_id))
        if cframe > segment[1]:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        templates = [
            ['in', in_templates],
            ['out', out_templates],
        ]

        kickback = process_templates(gray_frame, templates, kickback, cframe, fps)

    cap.release()


def process_templates(frame, g_templates, kickback, cframe, fps):
    i = 0
    j = 0
    if kickback[0] is None or kickback[1] is None:
        for g_template in g_templates:
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
                j += 1
            i += 1
    else:
        max_val = get_similarity_ratio(frame, g_templates[kickback[0]][1][kickback[1]])
        # skip next frame until we get under the kickback value again
        if max_val < kickback[2]:
            kickback[0] = None
            kickback[1] = None
            kickback[2] = 1
        else:
            print('skip')

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
