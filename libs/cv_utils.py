from typing import Tuple

import cv2
from cv2 import UMat, VideoCapture
from cv2 import CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT

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
