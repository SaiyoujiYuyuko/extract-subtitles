#!/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

from argparse import ArgumentParser, FileType

from cv2 import VideoCapture, UMat, imshow, rectangle
from cv2 import namedWindow, destroyWindow, WINDOW_NORMAL
from cv2 import setMouseCallback, waitKey
from cv2 import EVENT_LBUTTONDOWN, EVENT_LBUTTONUP

def cv2WaitKey(block_ms = 1) -> str:
  return chr(waitKey(block_ms) & 0xFF)

def guiSelectionUMat(mat: UMat, title = "Rect Selection (r:replot; c:OK)", box_color = (0, 0, 0xFF)) -> Tuple[Tuple[int, int], Tuple[int, int]]:
  ''' (left_top, right_down) '''
  p_lt, p_rd = None, None
  def onMouseEvent(event, x, y, flags, param):
    nonlocal p_lt, p_rd
    if event == EVENT_LBUTTONDOWN:
      p_lt = (x, y)
    elif event == EVENT_LBUTTONUP:
      p_rd = (x, y)
      rectangle(shot, p_lt, p_rd, box_color, thickness=2)
      imshow(title, shot)
  namedWindow(title, WINDOW_NORMAL)
  setMouseCallback(title, onMouseEvent)
  shot = mat.copy()
  while True:
    imshow(title, shot)
    key = cv2WaitKey(0)
    if key == 'r': shot = mat.copy()
    elif key == 'c': break
  destroyWindow(title)
  return (p_lt, p_rd)

def rectLtrd2Xywh(lt: Tuple[int, int], rd: Tuple[int, int]) -> Tuple[int, int, int, int]:
  x1, y1 = lt
  x2, y2 = rd
  return (x1, y1, x2-x1, y2-y1)

def selectCropRect(cap: VideoCapture, title = "Video (c:OK)") -> Tuple[int, int, int, int]:
  ''' position&size (x, y, w, h) '''
  unfinished, img = cap.read()
  while unfinished:
    imshow(title, img)
    if cv2WaitKey() == 'c':
      lt, rd = guiSelectionUMat(img)
      return rectLtrd2Xywh(lt, rd)
    unfinished, img = cap.read()
  return None

app = ArgumentParser("gui_crop_select", description="Interactive video crop rect selection")
app.add_argument("video", nargs="+", type=FileType("r"), help="video paths")

if __name__ == "__main__":
  cfg = app.parse_args()
  for path in map(lambda it: it.name, cfg.video):
    cap = VideoCapture(path)
    t = selectCropRect(cap)
    cap.release()
    print(f"{t[0:2]}{t[2:4]}")
