# Using manual gaze annotation tool

## Abstract

This is a manual gaze annotation tool. Use a video (`mp4` or `avi` for example). The tool will allow you to click
through the video on a frame by frame basis and annotate specific areas of interest. The output is saved in a
dictionary. The results can be downloaded as a csv file.

## Requirements

- `Python > 3.7`
- `pip install -r gazeclassify/tools/manual_annotation/requirements.txt`
    - Alternatively, install `streamlit` and `pandas` directly

## Solution

Run `streamlit run gazeclassify/tools/manual_annotation/main.py` and the annotation app will open in your browser

## Documentation of the workflow

- The script `main.py` contains a basic streamlit app to manually annotate a video.
- Define your own areas of interest:

```python
AOI = (
    "Other",
    "First AOI",
    "Second AOI",
    "Third AOI"
)
```
