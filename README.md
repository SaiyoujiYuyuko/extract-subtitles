# Subtitles Extraction

Extract key frames from [Amanpreet Walia](https://github.com/amanwalia92).

This project is used to extract subtitles from the video. First, the key frames is extracted from the video, and then the subtitle area of the frame picture is cropped, and the text is recognized by the OCR.

## Getting Started

### Install following dependences

- __OpenCV-Python__ (used for basic video processing e.g. read-frame-stream, crop, frame-diff, processing-gui)
- __PyTesseract__ (only use its `image_to_string(img, lang)`)
- NumPy (`smooth` filter) (find it [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy))
- SciPy (`signal.argrelextrema`)
- StrsimPy (`NormalizedLevenshtein` string similiarity)
- Matplotlib (draw frame differences stem plot)
- ProgressBar

Install missing dependences first using `pip install -r requirements.txt`

### Install Tesseract OCR

[Download](https://github.com/UB-Mannheim/tesseract/wiki) and (try) run it, select language support in `tesseract --list-lang` if you want.

### Run

```
λ python extract_subtitles.py <videopath>
```
## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details
