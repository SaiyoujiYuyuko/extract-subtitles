#!/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
from re import findall
from pathlib import Path
from os import remove
from progressbar import ProgressBar

import cv2
from cv2 import CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
from pytesseract import image_to_string

import numpy as np
from scipy.signal import argrelextrema

import matplotlib.pyplot as plot


class PatternType:
  def __init__(self, regex, transform = lambda x: x):
    self.regex, self.transform = regex, transform
  def __call__(self, string):
    return [self.transform(group) for group in findall(self.regex, string)]
  def __repr__(self): return f"PatternType({self.regex})"

def toMapper(transform):
  return lambda xs: [transform(x) for x in xs]

def zipWithNext(xs):
  assert len(xs) >= 2, f"{len(xs)} too short (< 2)"
  for i in range(1, len(xs)):
    yield (xs[i-1], xs[i])

def require(p, message):
  if not p: raise(ValueError(message))

def snakeSplit(text): return text.strip().split("_")
def titleCased(texts, sep=" "): return sep.join(map(str.capitalize, texts))

def printAttributes(fmt=lambda k, v: f"[{titleCased(snakeSplit(k))}] {v}", sep="\n", **kwargs):
  text = sep.join([fmt(k, v) for (k, v) in kwargs.items()])
  print(text)

def printResult(op):
  def _invoke(*args, **kwargs):
    res = op(*args, **kwargs)
    print(res); return res



def smooth(x, len_window, window = "hanning") -> np.array:
  supported_windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
  print("smooth", len(x), len_window)
  if len_window < 3: return x
  require(x.ndim == 1, "smooth only accepts 1 dimension arrays")
  require(x.size >= len_window, "input vector must >= window size")

  require(window in supported_windows, f"window must of {supported_windows}")

  s = np.r_[2 * x[0] - x[len_window:1:-1],
            x, 2 * x[-1] - x[-1:-len_window:-1]]
  w = getattr(np, window)(len_window) if window != "flat" else np.ones(len_window, "d")
  y = np.convolve(w / w.sum(), s, mode="same")
  return y[len_window -1 : -len_window +1]

def cv2NormalWin(title):
  cv2.namedWindow(title, cv2.WINDOW_NORMAL)

def cv2WaitKey(key_code, delay_ms = 1) -> bool:
  require(len(key_code) == 1, f"{repr(key_code)} must be single char")
  return cv2.waitKey(delay_ms) & 0xFF == ord(key_code)

def cv2VideoProps(cap: cv2.VideoCapture, props = (CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT)) -> (int, int, int, int):
  ''' (count, fps, width, height) '''
  return tuple(map(int, map(cap.get, props)))

class Frame:
  ''' Class to hold information about each frame '''
  def __init__(self, no, img, value):
    self.no, self.img, self.value = no, img, value

  def __lt__(self, other): return self.no < other.no
  def __eq__(self, other): return self.no == other.no

global app_cfg

def inFramesDir(name) -> str: return str(app_cfg.frames_dir/name)
filename_frame = lambda it: f"frame_{it.no}.jpg"

WIN_SUBTITLE_RECT = "Subtitle Rect"
WIN_LAST_FRAME = "Last Frame"

def postprocessArgs():
  if app_cfg.crop != None:
    ((x,y), (w,h)) = app_cfg.crop
    app_cfg.crop_cfg = ((y, y+h), (x, x+w))

def recognizeText(name) -> str:
  img = cv2.imread(inFramesDir(name))
  if app_cfg.crop != None:
    ((y, y_end), (x, x_end)) = app_cfg.crop_cfg
    croped_img = img[y:y_end, x:x_end]
    if app_cfg.crop_debug:
      cv2.imshow(WIN_SUBTITLE_RECT, croped_img)
      cv2.waitKey(1)
    return image_to_string(croped_img, app_cfg.lang)
  else:
    return image_to_string(img, app_cfg.lang)

@printResult
def relativeChange(a: float, b: float) -> float: return (b - a) / max(a, b)


def solveFrameDifferences(cap: cv2.VideoCapture, on_frame = lambda x: ()) -> (list, list):
  frames, frame_diffs = [], []

  index = 0
  prev_frame, curr_frame = None, None

  unfinished, img = cap.read()
  prev_frame = img #< initial (prev == curr)
  cv2NormalWin(WIN_LAST_FRAME)
  (n_frame, _, _, _) = cv2VideoProps(cap)
  progress = ProgressBar(maxval=n_frame).start()
  while unfinished:
    curr_frame = cv2.cvtColor(img, cv2.COLOR_BGR2LUV) #luv
    if curr_frame is not None: # and prev_frame is not None
      diff = cv2.absdiff(curr_frame, prev_frame) #< main logic goes here
      count = np.sum(diff)
      frame_diffs.append(count)
      frame = Frame(index, img, count)
      #frames.append(frame)
    on_frame(curr_frame)
    prev_frame = curr_frame
    index = index + 1
    progress.update(index)
    unfinished, img = cap.read()
    if app_cfg.crop_debug:
      cv2.imshow(WIN_LAST_FRAME, prev_frame) #< must have single name, to animate
      if cv2WaitKey('q'): break
  progress.finish()
  return (frames, frame_diffs)

def sortTopFramesDsc(frames, n_top_frames):
  ''' sort the list in descending order '''
  frames.sort(lambda it: it.value, reverse=True)
  for keyframe in frames[:n_top_frames]:
    name = filename_frame(keyframe)
    cv2.imwrite(inFramesDir(name), keyframe.frame)

def writeFramesThreshold(frames):
  for (a, b) in zipWithNext(frames):
    if relativeChange(np.float(a.value), np.float(b.value)) < app_cfg.thres: continue
    #print("prev_frame:"+str(frames[i-1].value)+"  curr_frame:"+str(frames[i].value))
    name = filename_frame(frames[i])
    cv2.imwrite(inFramesDir(name), frames[i].frame)

def ocrWithLocalMaxima(frames, frame_diffs, on_new_subtitle = print) -> np.array:
  diff_array = np.array(frame_diffs)
  sm_diff_array = smooth(diff_array, app_cfg.window_size)
  frame_indices = np.subtract(np.asarray(argrelextrema(sm_diff_array, np.greater))[0], 1)
  last_subtitle = ""
  for frame in map(frames.__getitem__, frame_indices):
    name = filename_frame(frame)
    cv2.imwrite(inFramesDir(name), frame.img)
    subtitle = recognizeText(name)
    if subtitle != last_subtitle:
      last_subtitle = subtitle #< Check for repeated subtitles 
      on_new_subtitle(frame.no, subtitle)
    remove(inFramesDir(name)) #< Delete recognized frame images
  return sm_diff_array

def drawPlot(diff_array):
  plot.figure(figsize=(40, 20))
  plot.locator_params(numticks=100)
  plot.stem(diff_array, use_line_collection=True)
  plot.savefig(inFramesDir("plot.png"))

app = ArgumentParser(prog="extract_subtitles", description="extract subtitles using OCR with frame difference algorithm")
apg = app.add_argument_group("basic workflow")
apg.add_argument("video", type=FileType("r"), help="source file to extract from")
apg.add_argument("-sort-top-dsc", metavar="n", nargs="?", type=int, help="make top frames and data descending")
apg.add_argument("-thres", metavar="x.x", nargs="?", type=float, help="fixed threshold value (float)")
apg.add_argument("-no-local-maxima", action="store_true", help="don't apply local maxima criteria")

'''
Using crop mode(crop out subtitles area) can greatly improve recognition accuracy,
but you need to manually adjust the crop area by modifying the value of cropper parameters(x, y, w, h).
To debug the appropriate value, set -crop-debug to show cropped result.
'''

apg1 = app.add_argument_group("misc settings")
regex_tuple = PatternType("\((\d+),(\d+)\)", toMapper(int))
apg1.add_argument("-crop", metavar="(x,y)(w,h)", type=regex_tuple, default=None, help="crop out subtitles area, improve recognition accuracy")
apg1.add_argument("--crop-debug", action="store_true", help="show cropped result if avaliable")

apg1.add_argument("-lang", type=str, default="chi_sim", help="OCR language for tesseract engine (tesseract --list-langs)")
apg1.add_argument("-draw-plot", action="store_true", help="draw plot for statics")
apg1.add_argument("-frames-dir", type=Path, default=Path("frames/"), help="directory to store the processed frames")
apg1.add_argument("-window-size", type=int, default=13, help="smoothing window size")

if __name__ == "__main__":
  app_cfg = app.parse_args()
  cfg = app_cfg
  postprocessArgs()
  printAttributes(
    video_path=cfg.video.name,
    frame_directory=cfg.frames_dir,
    subtitle_language=cfg.lang,
    crop=cfg.crop
  )
  print("Extracting key frames...")

  capture = cv2.VideoCapture(cfg.video.name)
  cap_props = cv2VideoProps(capture)
  printAttributes(video_props=cap_props)
  (frames, frame_diffs) = solveFrameDifferences(capture)
  capture.release()
  cv2.destroyAllWindows()
  if cfg.sort_top_dsc != None: sortTopFramesDsc(frames, cfg.sort_top_dsc)
  if cfg.thres != None: writeFramesThreshold(frames)
  diff_array = ocrWithLocalMaxima(frames, frame_diffs) if not cfg.no_local_maxima else None
  if cfg.draw_plot: drawPlot(diff_array)
