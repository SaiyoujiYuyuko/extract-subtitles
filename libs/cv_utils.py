from typing import Tuple
from pathlib import Path

import cv2
from cv2 import UMat, VideoCapture
from cv2 import CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT

import numpy as np
from numpy import array, convolve
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

from libs.fun_utils import require

_levenshtein = NormalizedLevenshtein()
def stringSimilarity(a: str, b: str) -> float:
  return _levenshtein.distance(a, b)

def relativeChange(a: float, b: float) -> float:
  return (b - a) / max(a, b)

# == OpenCV utils ==
def cv2NormalWin(title):
  cv2.namedWindow(title, cv2.WINDOW_NORMAL)

def cv2WaitKey(delay_ms = 1) -> str:
  return chr(cv2.waitKey(delay_ms) & 0xFF)

def cv2VideoProps(cap: VideoCapture) -> Tuple[int]:
  ''' (count, fps, width, height) '''
  props = (CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT)
  return tuple(map(int, map(cap.get, props)))

class Rect:
  def __init__(self, x,y, w,h):
    ''' (pad_left, pad_top, width, height) '''
    self.xywh = (x,y, w,h)
    self.form_range = (y,y+h, x,x+w)
  def sliceUMat(self, mat: UMat) -> UMat:
    (y, y_end, x, x_end) = self.form_range
    return mat[y:y_end, x:x_end]

class Frame:
  ''' Class to hold information about each frame '''
  def __init__(self, no, img, value):
    self.no, self.img, self.value = no, img, value
  def __eq__(self, other): return self.no == other.no
  def __hash__(self): return hash(self.no)


smooth_supported_windows = ["flat", "hanning", "hamming", "bartlett", "blackman"]
def smooth(a: array, window_size: int, window = "hanning") -> array:
  supported_windows = smooth_supported_windows
  if window_size < 3: return a
  require(a.ndim == 1, "smooth only accepts 1 dimension arrays")
  require(a.size >= window_size, "input vector size must >= window size")
  require(window in supported_windows, f"window must in {supported_windows}")

  s = np.r_[2 * a[0] - a[window_size:1:-1],
            a, 2 * a[-1] - a[-1:-window_size:-1]]
  w = getattr(np, window)(window_size) if window != "flat" else np.ones(window_size, "d")
  y = convolve(w / w.sum(), s, mode="same")
  return y[window_size -1 : -window_size +1]

class BasicCvProcess:
  ''' Helper class for simple CV programs (window+size, chunk_size, path_frames) '''
  def __init__(self, window: str, window_size: int, chunk_size: int, path_frames: Path):
    require(chunk_size > window_size, f"chunk size({chunk_size}) must fill(≥) window({window_size})")
    require(path_frames.is_dir(), f"{path_frames} must be dir")
    self.window, self.window_size, self.chunk_size, self.path_frames = window, window_size, chunk_size, path_frames
  @staticmethod
  def registerArguments(ap):
    ap.add_argument("--window", type=str, default="hamming", help=f"filter window, one of {smooth_supported_windows}")
    ap.add_argument("--window-size", type=int, default=30, help="matrix filtering window size")
    ap.add_argument("--chunk-size", type=int, default=300, help="processing frame chunk size")
    ap.add_argument("--frames-dir", type=Path, default=Path("frames/"), help="directory to store the processed frames")
  def frameFilepath(self, it: Frame) -> str:
    return str(self.path_frames/f"frame_{it.no}.jpg")
