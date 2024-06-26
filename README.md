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

### 2. Prepare templates

Template are image files that are similar to the frame that needs to be cut on the video.

There is template for the start and for the end.

You can add those images files into the directory `./files/templates`. You can add multiple template images in this directory.

You just need to prefix the filename with `in_` for starting template images and with `_out` for ending template images.

Exemple:
```
/files/templates
    in_template_1.png
    in_template_2.png
    out_template_1.png
```

You can set the `TEMPLATE_PATH` in the .env file to change the path to the template directory.

You can also set `TEMPLATE_IN_FILE_PATTERN` and `TEMPLATE_OUT_FILE_PATTERN` to change the templates filenames prefix.


### 3. Run

You can call `find_keyframes.py` to get the keyframes of the given file, and you can pipe the output directly
to `cut_on_keyframes.py` to cut the file using `ffmpeg`
```
python find_keyframes.py <path-to-file> --silent | python cut_on_keyframes.py --out <path-to-output-directory>
```

## Tune settings

### Process count

By default, the app will run on 4 process to run the app faster. You can change this by setting the `PROCESS_COUNT` env variable.

```env
PROCESS_COUNT=4
```

### Similarity ratio

The similarity ratio is a value used to compare how similar the template is with the frame that is currently being processed.

By default, it's set to `0.85`, but you can change it if needed. Value must be between `0` and `1`.

```env
SIMILARITY_RATIO=0.85
```

### Kickback ratio

The kickback ratio is a value used to skip all frames coming after that we found a frame that is matching our template.
This allows us to ignore all upcoming similar frames once we found one until the similarity ratio get down by the kickback ratio.

By default, it's set to `0.1`, but you can change it if needed. Value must be between `0` and `1`.

```env
KICKBACK_RATIO=0.1
```

### Skip frames count

This app do not process every single frame of the video otherwise it would take way too much time to process. 
By default, it will take only every `30` frames, but you can tune it by changing it in the `.env`

```
SKIP_FRAMES_COUNT=30
```