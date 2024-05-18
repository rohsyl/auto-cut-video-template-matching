# auto-cut-video-template-matching

Find timecodes where to cut a video automatically using template matching by cv2. Then use ffmpeg to cut them automatically.

## Requirements

- Python 3.x
- `cv2` library
- `dotenv` library
- `ffmpeg`

## Setup and Usage

### 1. Install dependencies

Install python dependencies
```
pip install opencv-python python-dotenv
```

Install ffmpeg and add it to the path

On Windows: 
```
choco install ffmpeg
```

On Ubuntu:
```
sudo apt install ffmpeg
```

> Make sure to add ffmpeg to the path or provide `FFMPEG_PATH` into the `.env` file

### 2. Run

You can call `find_keyframes.py` to get the keyframes of the given file, and you can pipe the output directly
to `cut_on_keyframes.py` to cut the file using `ffmpeg`
```
python find_keyframes.py <path-to-file> | python cut_on_keyframes.py
```