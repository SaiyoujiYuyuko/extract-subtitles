#!/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
from re import findall
from pathlib import Path
from os import remove
from progressbar import ProgressBar

import cv2
from cv2 import CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
import numpy as np
from numpy import array, convolve
from scipy.signal import argrelextrema
from pytesseract import image_to_string

import matplotlib.pyplot as plot

# == Normal function utils ==
identity = lambda x: x
noOp = lambda x: ()

def let(transform, x):
  return transform(x) if x != None else None

def also(op, x):
  op(x)
  return x

def require(p, message):
  if not p: raise ValueError(message)

class PatternType:
  def __init__(self, regex, transform = identity):
    self.regex, self.transform = regex, transform
  def __repr__(self): return f"PatternType({self.regex})"
  def __call__(self, text):
    groups = findall(self.regex, text)
    return list(map(self.transform, groups))


def toMapper(transform):
  return lambda xs: [transform(x) for x in xs]

def zipWithNext(xs: list):
  require(len(xs) >= 2, f"len {len(xs)} is too short (< 2)")
  for i in range(1, len(xs)):
    yield (xs[i-1], xs[i])

def snakeSplit(text): return text.strip().split("_")
def getFuncName(func): return findall("^<.*function (\S+)", repr(func))[0]
def titleCased(texts, sep = " "): return sep.join(map(str.capitalize, texts))

def printAttributes(fmt = lambda k, v: f"[{titleCased(snakeSplit(k))}] {v}", sep = "\n", **kwargs):
  entries = [fmt(k, v) for (k, v) in kwargs.items()]
  print(sep.join(entries))

printedCall_fmt = lambda op, args: f"{getFuncName(op)} {' '.join(map(str, args))}"
printedCall_on_result = lambda r: print("" if r == None else f" -> {r}")
def printedCall(op, fmt = printedCall_fmt, on_result = printedCall_on_result):
  def _invoke(*args, **kwargs):
    print(fmt(op, args), end="")
    res = op(*args, **kwargs)
    on_result(res)
    return res
  return _invoke

# == OpenCV utils ==
def cv2NormalWin(title):
  cv2.namedWindow(title, cv2.WINDOW_NORMAL)

def cv2WaitKey(key_code, delay_ms = 1) -> bool:
  require(len(key_code) == 1, f"{repr(key_code)} must be single char")
  return cv2.waitKey(delay_ms) & 0xFF == ord(key_code)

def cv2VideoProps(cap: cv2.VideoCapture) -> (int, int, int, int):
  ''' (count, fps, width, height) '''
  props = (CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT)
  return tuple(map(int, map(cap.get, props)))

def smooth(a, window_size, window = "hanning") -> array:
  supported_windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
  print(f"smooth [...x{len(a)}], {window_size} {window}")
  if window_size < 3: return a
  require(a.ndim == 1, "smooth only accepts 1 dimension arrays")
  require(a.size >= window_size, "input vector size must >= window size")

  require(window in supported_windows, f"window must in {supported_windows}")

  s = np.r_[2 * a[0] - a[window_size:1:-1],
            a, 2 * a[-1] - a[-1:-window_size:-1]]
  w = getattr(np, window)(window_size) if window != "flat" else np.ones(window_size, "d")
  y = np.convolve(w / w.sum(), s, mode="same")
  return y[window_size -1 : -window_size +1]

class Rect:
  def __init__(self, x,y, w,h):
    ''' (pad_left, pad_top, width, height) '''
    self.xywh = (x,y, w,h)
    self.form_range = (y,y+h, x,x+w)
  def sliceUMat(self, mat: cv2.UMat) -> cv2.UMat:
    (y, y_end, x, x_end) = self.form_range
    return mat[y:y_end, x:x_end]

# == App ==
class Frame:
  ''' Class to hold information about each frame '''
  def __init__(self, no, img, value):
    self.no, self.img, self.value = no, img, value
  def __eq__(self, other): return self.no == other.no
  def __hash__(self): return hash(self.no)

class ExtractSubtitles:
  WIN_SUBTITLE_RECT = "Subtitle Rect"
  WIN_LAST_FRAME = "Last Frame"

  def __init__(self, lang: str, is_crop_debug: bool, diff_thres: float, window_size: int, path_frames: Path):
    '''
    lang: language for Tesseract OCR
    is_crop_debug: show OpenCV capture GUI when processing
    diff_thres: threshold for differental frame dropper
    window_size: window size for numpy algorithms
    path_frames: temporary path for frame files
    '''
    require(path_frames.is_dir(), f"{path_frames} must be dir")
    self.lang, self.is_crop_debug = lang, is_crop_debug
    self.diff_thres, self.window_size, self.path_frames = diff_thres, window_size, path_frames
  def inFramesDir(self, name) -> str:
    return str(self.path_frames/name)
  def filename_frame(self, it): return f"frame_{it.no}.jpg"

  def recognizeText(self, name: str, crop: Rect) -> str:
    img = cv2.imread(self.inFramesDir(name))
    if crop != None:
      croped_img = crop.sliceUMat(img)
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_SUBTITLE_RECT, croped_img)
        cv2WaitKey(' ')
      return image_to_string(croped_img, self.lang)
    else:
      return image_to_string(img, self.lang)

  @printedCall
  @staticmethod
  def relativeChange(a: float, b: float) -> float: return (b - a) / max(a, b)

  def solveFrameDifferences(self, cap: cv2.VideoCapture, on_frame = noOp) -> (list, list):
    frames, frame_diffs = [], []

    index = 0
    prev_frame, curr_frame = None, None

    unfinished, img = cap.read()
    prev_frame = self.cvtColor(img) #< initial (prev == curr)
    if self.is_crop_debug: cv2NormalWin(ExtractSubtitles.WIN_LAST_FRAME)
    (n_frame, _, _, _) = cv2VideoProps(cap)
    progress = ProgressBar(maxval=n_frame).start()
    while unfinished:
      curr_frame = self.cvtColor(img)
      if curr_frame is not None: # and prev_frame is not None
        diff = cv2.absdiff(curr_frame, prev_frame) #< main logic goes here
        count = np.sum(diff)
        frame_diffs.append(count)
        frame = Frame(index, img, count)
        frames.append(frame)
      on_frame(curr_frame)
      prev_frame = curr_frame
      index = index + 1
      progress.update(index)
      unfinished, img = cap.read()
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_LAST_FRAME, prev_frame) #< must have single name, to animate
        if cv2WaitKey('q'): break
    progress.finish()
    return (frames, frame_diffs)

  def cvtColor(self, mat: cv2.UMat) -> cv2.UMat:
    return cv2.cvtColor(mat, cv2.COLOR_BGR2LUV)

  def writeFramesThreshold(self, frames):
    for (a, b) in zipWithNext(frames):
      if relativeChange(np.float(a.value), np.float(b.value)) < self.diff_thres: continue
      print(f"[{b.no}] prev: {a.value}, curr: {b.value}")
      name = self.filename_frame(frames[i])
      cv2.imwrite(self.inFramesDir(name), frames[i].img)

  def ocrWithLocalMaxima(self, frames, frame_diffs, crop, on_new_subtitle = print) -> np.array:
    diff_array = np.array(frame_diffs)
    sm_diff_array = smooth(diff_array, self.window_size)
    frame_indices = np.subtract(np.asarray(argrelextrema(sm_diff_array, np.greater))[0], 1)
    last_subtitle = ""
    for frame in map(frames.__getitem__, frame_indices):
      name = self.filename_frame(frame)
      cv2.imwrite(self.inFramesDir(name), frame.img)
      subtitle = self.recognizeText(name, crop)
      if not self.subtitleEquals(subtitle, last_subtitle):
        last_subtitle = subtitle #< Check for repeated subtitles 
        on_new_subtitle(frame.no, subtitle)
      remove(self.inFramesDir(name)) #< Delete recognized frame images
    return sm_diff_array

  def subtitleEquals(self, a, b) -> bool:
    return a == b

  @staticmethod
  def drawPlot(diff_array):
    plot.figure(figsize=(40, 20))
    plot.locator_params(numticks=100)
    plot.stem(diff_array, use_line_collection=True)
    plot.savefig(self.inFramesDir("plot.png"))

  def runOn(self, cap: cv2.VideoCapture, crop: Rect):
    '''
    cap: video input
    crop: Rect area for lyric graphics
    '''
    (frames, frame_diffs) = self.solveFrameDifferences(cap)
    cv2.destroyAllWindows()
    if self.diff_thres != None: writeFramesThreshold(frames)
    return self.ocrWithLocalMaxima(frames, frame_diffs, crop)


# == Main ==
app = ArgumentParser(
  prog="extract_subtitles",
  description="extract subtitles using OpenCV / Tesseract OCR with frame difference algorithm")

apg = app.add_argument_group("basic workflow")
apg.add_argument("video", type=FileType("r"), help="source file to extract from")
apg.add_argument("-crop", metavar="(x,y)(w,h)",
  type=PatternType("\((\d+),(\d+)\)", toMapper(int)),
  default=None, help="crop out subtitles area, improve recognition accuracy")
apg.add_argument("-thres", metavar="x.x", nargs="?", type=float, default=None, help="fixed threshold value (float)")
apg.add_argument("-lang", type=str, default="chi_sim", help="OCR language for tesseract engine (tesseract --list-langs)")

apg1 = app.add_argument_group("misc settings")
apg1.add_argument("--crop-debug", action="store_true", help="show OpenCV GUI when processing")
apg1.add_argument("--draw-plot", action="store_true", help="draw plot for statics")
apg1.add_argument("--window-size", type=int, default=13, help="smoothing window size")
apg1.add_argument("--frames-dir", type=Path, default=Path("frames/"), help="directory to store the processed frames")

def mkdirIfNotExists(self: Path):
  if not self.exists(): self.mkdir()

if __name__ == "__main__":
  cfg = app.parse_args()
  video_path = cfg.video.name
  lang, crop, crop_debug, thres, window_size, frames_dir = cfg.lang, cfg.crop, cfg.crop_debug, cfg.thres, cfg.window_size, cfg.frames_dir
  printAttributes(
    video_path=video_path,
    crop=crop,
    threshold=thres,
    subtitle_language=lang,
    frame_directory=frames_dir,
    filter_window_size=window_size
  )
  print("Extracting key frames...")
  capture = cv2.VideoCapture(video_path)
  cap_props = cv2VideoProps(capture)
  printAttributes(video_props=cap_props)
  extractor = ExtractSubtitles(lang, crop_debug, thres, window_size, also(mkdirIfNotExists, Path(frames_dir)))
  diff_array = extractor.runOn(capture, Rect(*crop[0], *crop[1]))
  capture.release()
  if cfg.draw_plot: ExtractSubtitles.drawPlot(diff_array)
