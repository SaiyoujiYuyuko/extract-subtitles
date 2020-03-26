#!/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
from re import findall
from pathlib import Path
from os import remove
from sys import stderr
from progressbar import ProgressBar
from itertools import chain, islice
from functools import reduce

import cv2
from cv2 import CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
import numpy as np
from numpy import array, convolve, concatenate
from scipy.signal import argrelextrema
from pytesseract import image_to_string
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

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

def chunked(n, xs):
  while True:
    try: #< must return when inner gen finished
      first = next(xs)
    except StopIteration: return
    chunk = islice(xs, n)
    yield chain((first,) , chunk)

def collect2(selector2, xs):
  bs, cs = [], []
  for x in xs:
    b, c = selector2(x)
    bs.append(b); cs.append(c)
  return (bs, cs)


def snakeSplit(text): return text.strip().split("_")
def titleCased(texts, sep = " "): return sep.join(map(str.capitalize, texts))

def printAttributes(fmt = lambda k, v: f"[{titleCased(snakeSplit(k))}] {v}", sep = "\n", **kwargs):
  entries = [fmt(k, v) for (k, v) in kwargs.items()]
  print(sep.join(entries))

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

class Rect:
  def __init__(self, x,y, w,h):
    ''' (pad_left, pad_top, width, height) '''
    self.xywh = (x,y, w,h)
    self.form_range = (y,y+h, x,x+w)
  def sliceUMat(self, mat: cv2.UMat) -> cv2.UMat:
    (y, y_end, x, x_end) = self.form_range
    return mat[y:y_end, x:x_end]

# == App ==
DEBUG = False
NOT_COMMON_PUNTUATION = "#$%&\\()*+-/:;<=>@[]^_`{|}" + "—»™€°"

class Frame:
  ''' Class to hold information about each frame '''
  def __init__(self, no, img, value):
    self.no, self.img, self.value = no, img, value
  def __eq__(self, other): return self.no == other.no
  def __hash__(self): return hash(self.no)

smooth_supported_windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
def smooth(a, window_size, window = "hanning") -> array:
  supported_windows = smooth_supported_windows
  if DEBUG: print(f"smooth [...x{len(a)}], {window_size} {window}")
  if window_size < 3: return a
  require(a.ndim == 1, "smooth only accepts 1 dimension arrays")
  require(a.size >= window_size, "input vector size must >= window size")
  require(window in supported_windows, f"window must in {supported_windows}")

  s = np.r_[2 * a[0] - a[window_size:1:-1],
            a, 2 * a[-1] - a[-1:-window_size:-1]]
  w = getattr(np, window)(window_size) if window != "flat" else np.ones(window_size, "d")
  y = np.convolve(w / w.sum(), s, mode="same")
  return y[window_size -1 : -window_size +1]

def stripAll(symbols, text) -> str:
  return text.translate({ord(c):"" for c in symbols})

class BasicCvProcess:
  ''' Helper class for simple CV programs (window+size, chunk_size, path_frames) '''
  def __init__(self, window: str, window_size: int, chunk_size: int, path_frames: Path):
    self.window, self.window_size, self.chunk_size, self.path_frames = window, window_size, chunk_size, path_frames
  @staticmethod
  def registerArguments(ap: ArgumentParser):
    ap.add_argument("--window", type=str, default="hamming", help=f"filter window, one of {smooth_supported_windows}")
    ap.add_argument("--window-size", type=int, default=30, help="smoothing window size")
    ap.add_argument("--chunk-size", type=int, default=300, help="processing frame chunk size")
    ap.add_argument("--frames-dir", type=Path, default=Path("frames/"), help="directory to store the processed frames")
  def frameFilepath(self, it: Frame) -> str:
    return str(self.path_frames/f"frame_{it.no}.jpg")

def relativeChange(a: float, b: float) -> float: return (b - a) / max(a, b)

class ExtractSubtitles(BasicCvProcess):
  WIN_LAST_IMAGE = "Last Image"
  WIN_LAST_FRAME = "Last Frame (processed image)"
  WIN_SUBTITLE_RECT = "Subtitle Rect"

  def __init__(self, lang: str, is_crop_debug: bool, diff_thres: float, window, window_size, chunk_size, path_frames):
    '''
    lang: language for Tesseract OCR
    is_crop_debug: show OpenCV capture GUI when processing
    diff_thres: threshold for differental frame dropper
    window: windowing kind
    window_size: window size for numpy algorithms
    chunk_size: processing chunk size for `ocrWithLocalMaxima()`
    path_frames: temporary path for frame files
    '''
    require(path_frames.is_dir(), f"{path_frames} must be dir")
    self.lang, self.is_crop_debug, self.diff_thres = lang, is_crop_debug, diff_thres
    super().__init__(window, window_size, chunk_size, path_frames)

  def cropUMat(self, mat: cv2.UMat, crop: Rect) -> cv2.UMat:
    if crop != None:
      croped_img = crop.sliceUMat(mat)
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_SUBTITLE_RECT, croped_img)
        cv2WaitKey(' ')
      return croped_img
    else: return mat

  def cvtUMatColor(self, mat: cv2.UMat) -> cv2.UMat:
    return mat # cv2.cvtColor(mat, cv2.COLOR_BGR2LUV)

  def solveFrameDifferences(self, cap: cv2.VideoCapture, crop: Rect) -> Frame:
    index = 0
    prev_frame, curr_frame = None, None
    postprocess = lambda mat: self.cvtUMatColor(self.cropUMat(mat, crop))

    unfinished, img = cap.read()
    prev_frame = postprocess(img) #< initial (prev == curr)
    if self.is_crop_debug:
      cv2NormalWin(ExtractSubtitles.WIN_LAST_IMAGE)
      cv2NormalWin(ExtractSubtitles.WIN_LAST_FRAME)
    (n_frame, _, _, _) = cv2VideoProps(cap)
    progress = ProgressBar(maxval=n_frame).start()
    while unfinished:
      curr_frame = postprocess(img)
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_LAST_IMAGE, img)
        cv2.imshow(ExtractSubtitles.WIN_LAST_FRAME, curr_frame) #< must have single title, to animate
        if cv2WaitKey('q'): break
      if curr_frame is not None: # and prev_frame is not None
        diff = cv2.absdiff(curr_frame, prev_frame) #< main logic goes here
        count = np.sum(diff)
        yield Frame(index, curr_frame, count)
      prev_frame = curr_frame
      unfinished, img = cap.read()
      index = index + 1
      progress.update(index)
    progress.finish()

  def writeFramesThresholded(self, frames):
    for (a, b) in zipWithNext(frames):
      vb = np.float(b.value if b.value != 0 else 1) #< what if no motion between (last-1)&last ?
      if relativeChange(np.float(a.value), vb) < self.diff_thres: continue
      if DEBUG: print(f"[{b.no}] prev: {a.value}, curr: {b.value}", file=stderr)
      cv2.imwrite(self.frameFilepath(a), a.img)

  def recognizeText(self, path) -> str:
    img = cv2.imread(path)
    return image_to_string(img, self.lang)
  
  def onFrameList(self, frames):
    if self.diff_thres != None: self.writeFramesThresholded(frames)

  def ocrWithLocalMaxima(self, frames, on_new_subtitle) -> array:
    ''' chunked processing using window, reducing memory usage '''
    frame_list, frame_diffs = collect2(lambda it: (it, it.value), frames)
    self.onFrameList(frame_list)
    diff_array = smooth(np.array(frame_diffs), self.window_size, self.window)
    frame_indices = np.subtract(np.asarray(argrelextrema(diff_array, np.greater))[0], 1)

    output_lose_subtitle = (self.path_frames/"loser.txt").open("a+")
    last_subtitle = ""
    for frame in map(frame_list.__getitem__, frame_indices):
      path = self.frameFilepath(frame)
      cv2.imwrite(path, frame.img)
      subtitle = self.recognizeText(path)
      remove(path) #< Delete recognized frame images
      if self.subtitleShouldReplace(last_subtitle, subtitle): #< check for repeated subtitles 
        last_subtitle = subtitle #v also clean-up new subtitle
        on_new_subtitle(frame.no, self.postprocessSubtitle(subtitle))
      else:
        output_lose_subtitle.write(f"{frame.no} {subtitle}\n")
    output_lose_subtitle.close()
    return diff_array

  _levenshtein = NormalizedLevenshtein()
  def stringSimilarity(self, a, b) -> float: return self._levenshtein.distance(a, b)

  def subtitleShouldReplace(self, a, b) -> bool:
    return b != a and b.count("\n") == 0 and self.stringSimilarity(a, b) > (1/4)

  def postprocessSubtitle(self, text) -> str:
    return stripAll(NOT_COMMON_PUNTUATION, text)

  def runOn(self, cap: cv2.VideoCapture, crop: Rect, on_new_subtitle = print) -> array:
    '''
    cap: video input
    crop: Rect area for lyric graphics
    '''
    frames = self.solveFrameDifferences(cap, crop)
    processChunk = lambda it: self.ocrWithLocalMaxima(it, on_new_subtitle)
    diff_array_parts = map(processChunk, chunked(self.chunk_size, frames))
    diff_array = reduce(lambda a, b: concatenate(array([a, b])), diff_array_parts)
    cv2.destroyAllWindows()
    return diff_array

  def drawPlot(self, diff_array):
    plot.figure(figsize=(40, 20))
    plot.locator_params(100)
    plot.stem(diff_array, use_line_collection=True)
    plot.savefig(self.path_frames/"plot.png")

# == Main ==
app = ArgumentParser(
  prog="extract_subtitles",
  description="Extract subtitles using OpenCV / Tesseract OCR with frame difference algorithm")

apg = app.add_argument_group("basic workflow")
apg.add_argument("video", type=FileType("r"), help="source file to extract from")
apg.add_argument("-crop", metavar="(x,y)(w,h)",
  type=PatternType("\((\d+),(\d+)\)", toMapper(int)),
  default=None, help="crop out subtitles area, improve recognition accuracy")
apg.add_argument("-thres", metavar="x.x", type=float, default=None, help="fixed threshold value")
apg.add_argument("-lang", type=str, default="eng", help="OCR language for Tesseract `tesseract --list-langs`")

apg1 = app.add_argument_group("misc settings")
apg1.add_argument("--crop-debug", action="store_true", help="show OpenCV GUI when processing")
apg1.add_argument("--draw-plot", action="store_true", help="draw difference plot for statics")
BasicCvProcess.registerArguments(apg1)

def mkdirIfNotExists(self: Path):
  if not self.exists(): self.mkdir()

if __name__ == "__main__":
  cfg = app.parse_args()
  video_path = cfg.video.name
  lang, crop, crop_debug, thres, window, window_size, chunk_size, frames_dir = cfg.lang, cfg.crop, cfg.crop_debug, cfg.thres, cfg.window, cfg.window_size, cfg.chunk_size, cfg.frames_dir
  printAttributes(
    video_path=video_path,
    subtitle_language=lang,
    crop=crop,
    threshold=thres,
    filter_window=window,
    filter_window_size=window_size,
    process_chunk_size=chunk_size,
    frame_directory=frames_dir
  )
  print("Extracting key frames...")
  capture = cv2.VideoCapture(video_path)
  cap_props = cv2VideoProps(capture)
  printAttributes(video_props=cap_props)
  extractor = ExtractSubtitles(lang, crop_debug, thres, window, window_size, chunk_size, also(mkdirIfNotExists, Path(frames_dir)))
  diff_array = extractor.runOn(capture, let(lambda it: Rect(*it[0], *it[1]), crop))
  capture.release()
  if cfg.draw_plot: extractor.drawPlot(diff_array)
