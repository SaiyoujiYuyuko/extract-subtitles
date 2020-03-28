# -*- coding: utf-8 -*-

from typing import Tuple, List, Iterator
from re import findall

SEC_MS = 1000
MIN_MS = 60*SEC_MS
HOUR_MS = 60*MIN_MS

def time_just(v: float, n = 2, pad = '0') -> str:
  text = str(int(v))
  return text.rjust(n, pad)

def millis2MinSecMs(ms) -> Tuple[int, int, int]:
  mins, r = divmod(ms, MIN_MS)
  secs, r = divmod(r, SEC_MS)
  return (mins, secs, r)

def millis2HourMinSecMs(ms) -> Tuple[int, int, int, int]:
  hrs, r = divmod(ms, HOUR_MS)
  mins, r = divmod(r, MIN_MS)
  secs, r = divmod(r, SEC_MS)
  return (hrs, mins, secs, r)

def makeConvertorFps2Ms(fps):
  ''' creates a frame-no to millis convertor '''
  return lambda no: (float(no) / fps) * SEC_MS


def millis2LrcTime(ms, ms_sep = ".") -> str:
  mins, secs, r = millis2MinSecMs(ms)
  return f"{time_just(mins)}:{time_just(secs)}{ms_sep}{int(r)}"

def dumpsLrc(ms, text) -> str:
  return f"[{millis2LrcTime(ms)}] {text}"

def loadsLrc(text) -> Tuple[int, str]:
  mm, ss, rrr, lyric = findall(r"^\[(\d{2}):(\d{2}).(\d+)\] ?(.+)$", text)[0]
  mins, secs, r = map(int, (mm, ss, rrr) )
  return (mins*MIN_MS + secs*SEC_MS + r, lyric)
