#!/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, List, Iterator

from argparse import ArgumentParser, Namespace, FileType
from pathlib import Path
from sys import argv, stderr
from functools import reduce
from progressbar import ProgressBar

from json import dumps

from libs.fun_utils import let, also, require
from libs.fun_utils import zipWithNext, chunked, collect2, expandRangeStartList
from libs.fun_utils import PatternType, toMapper, printAttributes
from libs.fun_utils import Reducer, AsNoOp

from libs.cv_utils import Frame, Rect, BasicCvProcess
from libs.cv_utils import smooth as orig_smooth, relativeChange, stringSimilarity
from libs.cv_utils import cv2VideoProps, cv2NormalWin, cv2WaitKey
from libs.cv_utils import cvScale, cvBlur, cvGrayscale

import cv2
from cv2 import UMat, VideoCapture
from pytesseract import image_to_string

import numpy as np
from numpy import array, concatenate
from scipy import signal

import matplotlib.pyplot as plot

# == App Common Logics ==
USE_FEATURE = set([])
FEAT_DEBUG = "--debug"
FEAT_PROGRESS = "--use-progress"
FEAT_SHARP = "--use-sharp"
NOT_COMMON_PUNTUATION = "#$%&\\()*+-/:;<=>@[]^_`{|}" + "—»™€°"

feats = USE_FEATURE.__contains__

def printDebug(*args, **kwargs):
  if feats(FEAT_DEBUG): print(*args, **kwargs, file=stderr)

def stripAll(symbols, text) -> str:
  return text.translate({ord(c):"" for c in symbols})

def smooth(a, window_size, window) -> array:
  printDebug(f"smooth [...x{len(a)}], {window_size} {window}")
  return orig_smooth(a, window_size, window)

def cvInGrayRange(img: UMat, start: int, end: int) -> UMat:
  return cv2.inRange(img, (start,start,start), (end,end,end))

class AsProgress(Reducer):
  def __init__(self, cap: VideoCapture, crop):
    n_frame = cv2VideoProps(cap)[0]
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
  - workflow: `runOn`, `solveFrameDifferences`, `findPeaks`, `onFrameList`, `ocrWithLocalMaxima`
  '''
  WIN_LAST_IMAGE = "Last Image"
  WIN_LAST_FRAME = "Last Frame (processed image)"
  WIN_SUBTITLE_RECT = "Subtitle Rect"

  def __init__(self, lang: str, is_crop_debug: bool, diff_save_thres: float, window, window_size, chunk_size, path_frames):
    '''
    - lang: language for Tesseract OCR
    - is_crop_debug: show OpenCV capture GUI when processing
    - diff_save_thres: save threshold for differential frame dropper
    - window: windowing kind
    - window_size: window size for numpy algorithms
    - chunk_size: processing chunk size for `ocrWithLocalMaxima()`
    - path_frames: temporary path for frame files
    '''
    self.lang, self.is_crop_debug, self.diff_save_thres = lang, is_crop_debug, diff_save_thres
    super().__init__(window, window_size, chunk_size, path_frames)

  def cropUMat(self, mat: UMat, crop: List[Rect], index: int) -> UMat:
    if crop == None: return mat
    cropped_img = crop[0].sliceUMat(mat)
    if self.is_crop_debug:
      cv2.imshow(ExtractSubtitles.WIN_SUBTITLE_RECT, cropped_img)
      cv2WaitKey()
    return cropped_img

  def postprocessUMat(self, mat: UMat, index: int) -> UMat: return mat

  def recognizeText(self, frame: Frame) -> str:
    return image_to_string(frame.img, self.lang)

  #v frame & subtitles
  def onFrameList(self, frames):
    if self.diff_save_thres != None: self.writeFramesThresholded(frames)

  def subtitleShouldReplace(self, a, b) -> bool:
    return b != a and b.count("\n") == 0 and stringSimilarity(a, b) > (1/4)

  def postprocessSubtitle(self, text) -> str:
    return stripAll(NOT_COMMON_PUNTUATION, text)

  def solveFrameDifferences(self, cap: VideoCapture, crop: List[Rect], fold) -> Iterator[Frame]:
    require(cap.isOpened(), "failed to open capture")
    postprocess = lambda mat, index: self.postprocessUMat(self.cropUMat(mat, crop, index), index)
    if self.is_crop_debug:
      cv2NormalWin(ExtractSubtitles.WIN_LAST_IMAGE)
      cv2NormalWin(ExtractSubtitles.WIN_LAST_FRAME)
    reducer = fold(cap, crop)

    index = 0
    prev_frame, curr_frame = None, None
    unfinished, img = cap.read()
    prev_frame = postprocess(img, 0) #< initial (prev == curr)
    while unfinished:
      curr_frame = postprocess(img, index)
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_LAST_IMAGE, img)
        cv2.imshow(ExtractSubtitles.WIN_LAST_FRAME, curr_frame) #< must have single title, to animate
        if cv2WaitKey() == 'q': break
      if curr_frame is not None: # and prev_frame is not None
        try:
          diff = cv2.absdiff(curr_frame, prev_frame) #< main algorithm goes here
          yield Frame(index, curr_frame, np.sum(diff))
        except cv2.error: pass
      prev_frame = curr_frame
      unfinished, img = cap.read()
      index = index + 1
      reducer.accept(index)
    reducer.finish()

  def postprocessDifferences(self, a: array) -> array: return smooth(a, self.window_size, self.window)
  def findPeaks(self, a: array) -> array: return np.asarray(signal.argrelextrema(a, np.greater))[0] #< argrelextrema(_) always (x,)

  def ocrWithLocalMaxima(self, frames, reducer) -> Tuple[array, array]:
    '''
    - frames: chunked processing using window, reducing memory usage
    - reducer: accept (frame, subtitle)
    '''
    frame_list, frame_diffs = collect2(lambda it: (it, it.value), frames)
    self.onFrameList(frame_list)

    diff_array = self.postprocessDifferences(array(frame_diffs))
    valid_indices = self.findPeaks(diff_array)
    for frame in map(frame_list.__getitem__, valid_indices):
      if self.is_crop_debug:
        cv2.imshow(ExtractSubtitles.WIN_SUBTITLE_RECT, frame.img)
        cv2WaitKey()
      subtitle = self.recognizeText(frame)
      reducer.accept(frame, subtitle)
    reducer.finish()
    return (diff_array, valid_indices)

  class DefaultOcrFold(Reducer):
    def __init__(self, ctx, name, on_new_subtitle = print):
      self.ctx = ctx; self.on_new_subtitle = on_new_subtitle
      self.files = [(ctx.path_frames/f"{group}_{name}.txt").open("a+") for group in ["timeline", "loser"]]
      self.out_timeline, self.out_lose_subtitle = self.files
      self.last_subtitle = ""
      self.frame_index = 0
    def accept(self, frame, subtitle):
      self.out_timeline.write(f"{frame.no} {dumps(subtitle, ensure_ascii=False)}\n")
      if self.ctx.subtitleShouldReplace(self.last_subtitle, subtitle): #< check for repeated subtitles 
        self.last_subtitle = subtitle #v also clean-up new subtitle
        self.on_new_subtitle(frame.no, self.ctx.postprocessSubtitle(subtitle))
      else:
        self.out_lose_subtitle.write(f"{frame.no} {subtitle}\n")
      if self.ctx.is_crop_debug:
        cv2.imwrite(str(self.ctx.path_frames/f"subtitle_{self.frame_index}.png"), frame.img)
      self.frame_index += 1
    def finish(self): #< in (single chunk) OCR
      for f in self.files: f.flush()
    def finishAll(self):
      for f in self.files: f.close()

  def runOn(self, cap: VideoCapture, crop: List[Rect], fold = DefaultOcrFold, name = "default") -> Tuple[array, array]:
    '''
    - cap: video input
    - crop: Rect area for lyric graphics
    - fold: init (self, name)
    '''
    frames = self.solveFrameDifferences(cap, crop, AsProgress if feats(FEAT_PROGRESS) else AsNoOp)
    reducer = fold(self, name)
    processChunk = lambda it: self.ocrWithLocalMaxima(it, reducer)
    diff_array_parts = map(processChunk, chunked(self.chunk_size, frames))
    def concatResults(a, b) -> Tuple[array, array]:
      a0, a1 = a
      b0, b1 = b
      ab0 = concatenate(array([a0, b0]))
      ab1 = concatenate(array([a1, b1+len(a0)]))
      return (ab0, ab1)
    (diff_array, indices) = reduce(concatResults, diff_array_parts)

    reducer.finishAll()
    cv2.destroyAllWindows()
    return (diff_array, indices)

  def writeFramesThresholded(self, frames):
    for (a, b) in zipWithNext(frames):
      if b.value == 0: continue #< what if no motion between (last-1)&last ?
      k_change = relativeChange(np.float(a.value), np.float(b.value))
      if k_change < self.diff_save_thres: continue
      printDebug(f"[{b.no}]({k_change}) prev: {a.value}, curr: {b.value}")
      cv2.imwrite(self.frameFilepath(a), a.img)

  def drawPlot(self, diff_array, indices):
    fig_diff = plot.figure(figsize=(40, 20))
    plot.xlabel("Frame.no")
    plot.ylabel("differences")
    plot.locator_params(100)
    plot.stem(diff_array, linefmt=":", use_line_collection=True)
    plot.stem(indices, [diff_array[i] for i in indices], use_line_collection=True)
    return fig_diff


# == Main ==
def makeArgumentParser():
  app = ArgumentParser(
    prog="extract_subtitles",
    description="Extract subtitles using OpenCV / Tesseract OCR with frame difference algorithm")

  apg = app.add_argument_group("basic workflow")
  apg.add_argument("video", nargs="+", type=FileType("r"), help="source file to extract from")
  apg.add_argument("-crop", metavar="frame(x,y)[w,h]",
    type=PatternType(r"(\d+)\((\d+),(\d+)\)\[(\d+),(\d+)\]", toMapper(int)),
    default=None, help="crop out subtitles area, improve recognition accuracy")
  apg.add_argument("-filter-code", type=str, default="it", help="(it: cv2.UMat) pipe function")
  apg.add_argument("-lang", type=str, default="eng", help="OCR language for Tesseract `tesseract --list-langs`")
  apg.add_argument("-save-thres", metavar="x.x", type=float, default=None, help="add frame store for fixed save threshold value")

  apg1 = app.add_argument_group("misc settings")
  apg1.add_argument("--crop-debug", action="store_true", help="show OpenCV GUI when processing")
  apg1.add_argument("--draw-plot", action="store_true", help="draw difference plot for statics")
  apg1.add_argument(FEAT_SHARP, action="store_true", help="use non-smooth differential (improve for timeline, slower)")
  apg1.add_argument(FEAT_PROGRESS, action="store_true", help="show progress bar")
  apg1.add_argument(FEAT_DEBUG, action="store_true", help="print debug info")
  BasicCvProcess.registerArguments(apg1)
  return app

def mkdirIfNotExists(self: Path):
  if not self.exists(): self.mkdir()

def makeExtractor(cfg: Namespace, cls_extract=ExtractSubtitles) -> ExtractSubtitles:
  lang, crop, crop_debug, save_thres, window, window_size, chunk_size, frames_dir = cfg.lang, cfg.crop, cfg.crop_debug, cfg.save_thres, cfg.window, cfg.window_size, cfg.chunk_size, cfg.frames_dir
  printAttributes(
    subtitle_language=lang,
    crop=crop,
    save_threshold=save_thres,
    filter_window=window,
    filter_window_size=window_size,
    process_chunk_size=chunk_size,
    frame_directory=frames_dir
  )
  if cfg.use_sharp: #< assign extra config
    USE_FEATURE.add(FEAT_SHARP)
  if cfg.use_progress:
    USE_FEATURE.add(FEAT_PROGRESS)
  if cfg.debug:
    USE_FEATURE.add(FEAT_DEBUG)

  extractor = cls_extract(lang, crop_debug, save_thres,
    window, window_size, chunk_size, also(mkdirIfNotExists, Path(frames_dir)) )
  return extractor

class EvalFilterExtractSubtitle(ExtractSubtitles):
  def __init__(self, *args, filter_code = "it"):
    ''' filter_code: Python expr about `(it: cv2.UMat)` results `cv2.UMat` '''
    super().__init__(*args)
    self.mat_filter = eval(compile(f"lambda it, i: {filter_code}", "<frame_filter>", "eval"))
    self.is_sharp = feats(FEAT_SHARP)
  def postprocessUMat(self, mat, index):
    return self.mat_filter(mat, index)
  def postprocessDifferences(self, a) -> array:
    return (a if self.is_sharp else super().postprocessDifferences(a))

class CropEvalFilterExtractSubtitle(EvalFilterExtractSubtitle):
  def cropUMat(self, mat, crop, index) -> UMat:
    cropped_img = crop[index].sliceUMat(mat)
    if self.is_crop_debug:
      cv2.imshow(ExtractSubtitles.WIN_SUBTITLE_RECT, cropped_img)
      cv2WaitKey()
    return cropped_img

# == Entry ==
def main(args):
  app = makeArgumentParser()
  cfg = app.parse_args(args)
  cls_extract = lambda *args: (EvalFilterExtractSubtitle if cfg.crop == None or len(cfg.crop) <= 1 else CropEvalFilterExtractSubtitle) (*args, filter_code=cfg.filter_code)
  extractor = makeExtractor(cfg, cls_extract=cls_extract)
  for video_path in map(lambda it: it.name, cfg.video):
    video_name = Path(video_path).name
    printAttributes(video_path=video_path)
    print("Extracting key frames...")

    capture = VideoCapture(video_path)
    n_frames, fps, w, h = cv2VideoProps(capture)
    printAttributes(video_playback=(n_frames, fps), video_dimens=(w, h))

    #v [(t, x,y, w,h), ...]
    key = lambda it: it[0]; makeRect = lambda it: Rect(*it[1:])
    crops = let(lambda t: [makeRect(t[0])] if len(t) == 1 else expandRangeStartList(n_frames, t, key=key, value=makeRect), cfg.crop)
    if crops != None: require(crops[0] != None, "first crop area must started at frame 0") #< only when multi-crop enabled
    (diff_array, indices) = extractor.runOn(capture, crops, name=video_name)
    capture.release()
    if cfg.draw_plot:
      fig_diff = extractor.drawPlot(diff_array, indices)
      print(indices)
      plot.title(f"Filtered differential sum for {video_name}")
      plot.show()
      fig_diff.savefig(cfg.frames_dir/f"plot_{video_name}.png")

if __name__ == "__main__": main(argv[1:]) #< no program name
