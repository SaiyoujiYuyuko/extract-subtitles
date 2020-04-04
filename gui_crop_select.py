#!/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Iterator, cast

from argparse import ArgumentParser, FileType

from cv2 import VideoCapture, UMat, imshow, rectangle
from cv2 import namedWindow, destroyWindow, WINDOW_NORMAL
from cv2 import setMouseCallback, waitKey
from cv2 import EVENT_LBUTTONDOWN, EVENT_LBUTTONUP
from cv2 import CAP_PROP_POS_FRAMES

def cv2WaitKey(block_ms = 1) -> str:
  return chr(waitKey(block_ms) & 0xFF)

def rectLtrd2Xywh(lt: Tuple[int, int], rd: Tuple[int, int]) -> Tuple[int, int, int, int]:
  x1, y1 = lt
  x2, y2 = rd
  return (x1, y1, x2-x1, y2-y1)

def cv2Crop(mat: UMat, ltrd) -> UMat:
  x,y, w,h = rectLtrd2Xywh(*ltrd)
  return mat[y:y+h, x:x+w]

# == App ==
def guiSelectionUMat(mat: UMat, title="Rect Selection (r:replot; c:OK)", box_color=(0xFF, 0x00, 0x00), thickness=3) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
  ''' (left_top, right_down) '''
  p_lt, p_rd = None, None
  shot = mat.copy()
  def onMouseEvent(event, x, y, flags, param):
    nonlocal p_lt, p_rd, shot
    if event == EVENT_LBUTTONDOWN:
      p_lt = (x, y)
    elif event == EVENT_LBUTTONUP:
      p_rd = (x, y)
      rectangle(shot, p_lt, p_rd, box_color, thickness)
      imshow(title, shot)
  namedWindow(title, WINDOW_NORMAL)
  setMouseCallback(title, onMouseEvent)
  imshow(title, shot)
  while True:
    key = cv2WaitKey(0)
    if key == 'c' and (p_lt != None and p_rd != None):
      destroyWindow(title)
      return cast(Tuple, (p_lt, p_rd))
    elif key == 'r':
      shot = mat.copy()
      imshow(title, shot)
    elif key == 'q':
      return None

def selectCropRects(cap: VideoCapture, title = "Video (c:OK; q:finished, <:-; >:=)", title_preview = "Preview", d_seek=5) -> Iterator[Tuple[int, Tuple[int, int, int, int]]]:
  ''' position&size (x, y, w, h) '''
  index = 0
  ltrd = None
  def seek(n):
    nonlocal index
    index += n
    cap.set(CAP_PROP_POS_FRAMES, index)
  def handleSeek(key):
    if key == '-': seek(-d_seek)
    elif key == '=': seek(+d_seek)
  def handleSelect():
    nonlocal ltrd
    area = guiSelectionUMat(img)
    if area != None:
      ltrd = area
      return (index, rectLtrd2Xywh(*ltrd))
  frame_ops = [lambda: seek(-1), lambda: seek(+1)]

  unfinished, img = cap.read()
  while unfinished:
    imshow(title, img)
    if ltrd != None: imshow(title_preview, cv2Crop(img, ltrd))
    key = cv2WaitKey()
    if key == 'c':
      select = handleSelect()
      if select != None: yield select
    elif key == 'q': break

    elif key in '-=': handleSeek(key)
    elif key == ' ':
      while True:
        key1 = cv2WaitKey(0)
        if key1 == ' ': break
        elif key1 == 'c':
          select = handleSelect()
          if select != None: yield select          
        elif key1 in "-=89":
          handleSeek(key1)
          miniseek = ord(key1) - ord('8') #[89] to mimiseek
          if miniseek in range(len(frame_ops)): frame_ops[miniseek]()
          unfinished, img = cap.read()
          imshow(title, img)
    unfinished, img = cap.read()
    index += 1

app = ArgumentParser("gui_crop_select", description="Interactive video crop rect selection")
app.add_argument("video", nargs="+", type=FileType("r"), help="video paths")

if __name__ == "__main__":
  cfg = app.parse_args()
  for path in map(lambda it: it.name, cfg.video):
    cap = VideoCapture(path)
    crops = selectCropRects(cap)
    for i, c in crops:
      (x,y, w,h) = c
      print(f"{i}({x},{y})[{w},{h}]")
    cap.release()
