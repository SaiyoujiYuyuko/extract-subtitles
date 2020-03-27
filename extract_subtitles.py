#!/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterator

from argparse import ArgumentParser, Namespace, FileType
from pathlib import Path
from os import remove
from sys import argv, stderr
from functools import reduce
from progressbar import ProgressBar

from json import dumps

from libs.fun_utils import let, also, require
from libs.fun_utils import zipWithNext, chunked, collect2
from libs.fun_utils import PatternType, toMapper, printAttributes
from libs.fun_utils import Reducer, AsNoOp

from libs.cv_utils import Frame, Rect
from libs.cv_utils import cv2VideoProps, cv2NormalWin, cv2WaitKey

import cv2
from cv2 import UMat, VideoCapture
from pytesseract import image_to_string

import numpy as np
from numpy import array, convolve, concatenate
from scipy.signal import argrelextrema
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

import matplotlib.pyplot as plot

# == App Common Logics ==
USE_FEATURE = set([])
FEAT_DEBUG = "--debug"
FEAT_PROGRESS = "--use-progress"
NOT_COMMON_PUNTUATION = "#$%&\\()*+-/:;<=>@[]^_`{|}" + "—»™€°"

feats = USE_FEATURE.__contains__

def printDebug(*args, **kwargs):
  if feats(FEAT_DEBUG): print(*args, **kwargs, file=stderr)

def relativeChange(a: float, b: float) -> float:
  return (b - a) / max(a, b)

smooth_supported_windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
def smooth(a: array, window_size: int, window = "hanning") -> array:
  supported_windows = smooth_supported_windows
  printDebug(f"smooth [...x{len(a)}], {window_size} {window}")
  if window_size < 3: return a
  require(a.ndim == 1, "smooth only accepts 1 dimension arrays")
  require(a.size >= window_size, "input vector size must >= window size")
  require(window in supported_windows, f"window must in {supported_windows}")

  s = np.r_[2 * a[0] - a[window_size:1:-1],
            a, 2 * a[-1] - a[-1:-window_size:-1]]
  w = getattr(np, window)(window_size) if window != "flat" else np.ones(window_size, "d")
  y = convolve(w / w.sum(), s, mode="same")
  return y[window_size -1 : -window_size +1]

def stripAll(symbols, text) -> str:
  return text.translate({ord(c):"" for c in symbols})

_levenshtein = NormalizedLevenshtein()
def stringSimilarity(a: str, b: str) -> float:
  return _levenshtein.distance(a, b)


class BasicCvProcess:
  ''' Helper class for simple CV programs (window+size, chunk_size, path_frames) '''
  def __init__(self, window: str, window_size: int, chunk_size: int, path_frames: Path):
    self.window, self.window_size, self.chunk_size, self.path_frames = window, window_size, chunk_size, path_frames
  @staticmethod
  def registerArguments(ap):
    ap.add_argument("--window", type=str, default="hamming", help=f"filter window, one of {smooth_supported_windows}")
    ap.add_argument("--window-size", type=int, default=30, help="matrix filtering window size")
    ap.add_argument("--chunk-size", type=int, default=300, help="processing frame chunk size")
    ap.add_argument("--frames-dir", type=Path, default=Path("frames/"), help="directory to store the processed frames")
  def frameFilepath(self, it: Frame) -> str:
    return str(self.path_frames/f"frame_{it.no}.jpg")

class AsProgress(Reducer):
  def __init__(self, cap: VideoCapture, crop):
    n_frame = int(cv2VideoProps(cap)[0])
    self.progress = ProgressBar(maxval=n_frame).start()
  def accept(self, index):
    self.progress.update(index)
  def finish(self):
    self.progress.finish()

# == Main Algorithm ==
class ExtractSubtitles(BasicCvProcess):
  '''
  Operation of extracting video subtitle area as text,
  - configurable: `cropUMat`, `postprocessUMat`, `onFrameList`, `subtitleShouledReplace`, `postpreocessSubtitle`
  - workflow: `runOn`, `solveFrameDifferences`, `onFrameList`, `ocrWithLocalMaxima`
  '''
  WIN_LAST_IMAGE = "Last Image"
  WIN_LAST_FRAME = "Last Frame (processed image)"
  WIN_SUBTITLE_RECT = "Subtitle Rect"

  def __init__(self, lang: str, is_crop_debug: bool, diff_thres: float, window, window_size, chunk_size, path_frames):
    '''
    - lang: language for Tesseract OCR
    - is_crop_debug: show OpenCV capture GUI when processing
    - diff_thres: threshold for differental frame dropper
    - window: windowing kind
    - window_size: window size for numpy algorithms
    - chunk_size: processing chunk size for `ocrWithLocalMaxima()`
    - path_frames: temporary path for frame files
    '''
    require(path_frames.is_dir(), f"{path_frames} must be dir")
    self.lang, self.is_crop_debug, self.diff_thres = lang, is_crop_debug, diff_thres
    super().__init__(window, window_size, chunk_size, path_frames)

  def cropUMat(self, mat: UMat, crop: Rect) -> UMat:
    if crop != None:
      croped_img = crop.sliceUMat(mat)
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_SUBTITLE_RECT, croped_img)
        cv2WaitKey()
      return croped_img
    else: return mat

  def postprocessUMat(self, mat: UMat) -> UMat:
    return mat #cv2.cvtColor(mat, cv2.COLOR_BGR2LUV)

  def solveFrameDifferences(self, cap: VideoCapture, crop: Rect, fold) -> Iterator[Frame]:
    postprocess = lambda mat: self.postprocessUMat(self.cropUMat(mat, crop))
    if self.is_crop_debug:
      cv2NormalWin(ExtractSubtitles.WIN_LAST_IMAGE)
      cv2NormalWin(ExtractSubtitles.WIN_LAST_FRAME)
    reducer = fold(cap, crop)

    index = 0
    prev_frame, curr_frame = None, None
    unfinished, img = cap.read()
    prev_frame = postprocess(img) #< initial (prev == curr)
    while unfinished:
      curr_frame = postprocess(img)
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_LAST_IMAGE, img)
        cv2.imshow(ExtractSubtitles.WIN_LAST_FRAME, curr_frame) #< must have single title, to animate
        if cv2WaitKey() == 'q': break
      if curr_frame is not None: # and prev_frame is not None
        diff = cv2.absdiff(curr_frame, prev_frame) #< main logic goes here
        count = np.sum(diff)
        yield Frame(index, curr_frame, count)
      prev_frame = curr_frame
      unfinished, img = cap.read()
      index = index + 1
      reducer.accept(index)
    reducer.finish()

  def writeFramesThresholded(self, frames):
    for (a, b) in zipWithNext(frames):
      vb = np.float(b.value if b.value != 0 else 1) #< what if no motion between (last-1)&last ?
      if relativeChange(np.float(a.value), vb) < self.diff_thres: continue
      printDebug(f"[{b.no}] prev: {a.value}, curr: {b.value}")
      cv2.imwrite(self.frameFilepath(a), a.img)

  def recognizeText(self, frame: Frame) -> str:
    #path = self.frameFilepath(frame)
    #cv2.imwrite(path, frame.img)
    #img = cv2.imread(path)
    #remove(path) #< Delete recognized frame images
    return image_to_string(frame.img, self.lang)

  def onFrameList(self, frames):
    if self.diff_thres != None: self.writeFramesThresholded(frames)

  def ocrWithLocalMaxima(self, frames, reducer) -> array:
    '''
    - frames: chunked processing using window, reducing memory usage
    - reducer: accept (frame, subtitle)
    '''
    frame_list, frame_diffs = collect2(lambda it: (it, it.value), frames)
    self.onFrameList(frame_list)
    diff_array = smooth(array(frame_diffs), self.window_size, self.window)
    frame_indices = np.subtract(np.asarray(argrelextrema(diff_array, np.greater))[0], 1)

    for frame in map(frame_list.__getitem__, frame_indices):
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_SUBTITLE_RECT, frame.img)
        cv2WaitKey()
      subtitle = self.recognizeText(frame)
      reducer.accept(frame, subtitle)
    reducer.finish()
    return diff_array

  class DefaultOcrFold(Reducer):
    def __init__(self, ctx, name, on_new_subtitle = print):
      self.ctx = ctx; self.on_new_subtitle = on_new_subtitle
      self.files = [(ctx.path_frames/f"{group}_{name}.txt").open("a+") for group in ["timeline", "loser"]]
      self.out_timeline, self.out_lose_subtitle = self.files
      self.last_subtitle = ""
    def accept(self, frame, subtitle):
      self.out_timeline.write(f"{frame.no} {dumps(subtitle, ensure_ascii=False)}\n")
      if self.ctx.subtitleShouldReplace(self.last_subtitle, subtitle): #< check for repeated subtitles 
        self.last_subtitle = subtitle #v also clean-up new subtitle
        self.on_new_subtitle(frame.no, self.ctx.postprocessSubtitle(subtitle))
      else:
        self.out_lose_subtitle.write(f"{frame.no} {subtitle}\n")
    def finish(self): #< in (single chunk) OCR
      for f in self.files: f.flush()
    def finishAll(self):
      for f in self.files: f.close()

  def subtitleShouldReplace(self, a, b) -> bool:
    return b != a and b.count("\n") == 0 and stringSimilarity(a, b) > (1/4)

  def postprocessSubtitle(self, text) -> str:
    return stripAll(NOT_COMMON_PUNTUATION, text)

  def runOn(self, cap: VideoCapture, crop: Rect, fold = DefaultOcrFold, name = "default") -> array:
    '''
    - cap: video input
    - crop: Rect area for lyric graphics
    - fold: init (self, name)
    '''
    frames = self.solveFrameDifferences(cap, crop, AsProgress if feats(FEAT_PROGRESS) else AsNoOp)
    reducer = fold(self, name)
    processChunk = lambda it: self.ocrWithLocalMaxima(it, reducer)
    diff_array_parts = map(processChunk, chunked(self.chunk_size, frames))
    diff_array = reduce(lambda a, b: concatenate(array([a, b])), diff_array_parts)

    reducer.finishAll()
    cv2.destroyAllWindows()
    return diff_array

  def drawPlot(self, diff_array):
    fig_diff = plot.figure(figsize=(40, 20))
    plot.locator_params(100)
    plot.stem(diff_array, use_line_collection=True)
    return fig_diff


# == Main ==
def makeArgumentParser():
  app = ArgumentParser(
    prog="extract_subtitles",
    description="Extract subtitles using OpenCV / Tesseract OCR with frame difference algorithm")

  apg = app.add_argument_group("basic workflow")
  apg.add_argument("video", nargs="+", type=FileType("r"), help="source file to extract from")
  apg.add_argument("-crop", metavar="(x,y)(w,h)",
    type=PatternType(r"\((\d+),(\d+)\)", toMapper(int)),
    default=None, help="crop out subtitles area, improve recognition accuracy")
  apg.add_argument("-thres", metavar="x.x", type=float, default=None, help="add frame store for fixed threshold value")
  apg.add_argument("-lang", type=str, default="eng", help="OCR language for Tesseract `tesseract --list-langs`")

  apg1 = app.add_argument_group("misc settings")
  apg1.add_argument("--crop-debug", action="store_true", help="show OpenCV GUI when processing")
  apg1.add_argument("--draw-plot", action="store_true", help="draw difference plot for statics")
  apg1.add_argument(FEAT_PROGRESS, action="store_true", help="show progress bar")
  apg1.add_argument(FEAT_DEBUG, action="store_true", help="print debug info")
  BasicCvProcess.registerArguments(apg1)
  return app

def mkdirIfNotExists(self: Path):
  if not self.exists(): self.mkdir()

def makeExtractor(cfg: Namespace):
  lang, crop, crop_debug, thres, window, window_size, chunk_size, frames_dir = cfg.lang, cfg.crop, cfg.crop_debug, cfg.thres, cfg.window, cfg.window_size, cfg.chunk_size, cfg.frames_dir
  printAttributes(
    subtitle_language=lang,
    crop=crop,
    threshold=thres,
    filter_window=window,
    filter_window_size=window_size,
    process_chunk_size=chunk_size,
    frame_directory=frames_dir
  )
  extractor = ExtractSubtitles(lang, crop_debug, thres,
    window, window_size, chunk_size, also(mkdirIfNotExists, Path(frames_dir)) )
  if cfg.use_progress: #< assign extra config
    USE_FEATURE.add(FEAT_PROGRESS)
  if cfg.debug:
    USE_FEATURE.add(FEAT_DEBUG)
  return extractor

# == Entry ==
def main(args):
  app = makeArgumentParser()
  cfg = app.parse_args(args)
  extractor = makeExtractor(cfg)
  for video_path in map(lambda it: it.name, cfg.video):
    video_name = Path(video_path).name
    printAttributes(video_path=video_path)
    print("Extracting key frames...")

    capture = VideoCapture(video_path)
    printAttributes(video_props=cv2VideoProps(capture))
    diff_array = extractor.runOn(capture, let(lambda it: Rect(*it[0], *it[1]), cfg.crop), name=video_name)
    capture.release()
    if cfg.draw_plot:
      fig_diff = extractor.drawPlot(diff_array)
      plot.show()
      fig_diff.savefig(cfg.frames_dir/f"plot_{video_name}.png")

if __name__ == "__main__": main(argv[1:]) #< no program name
