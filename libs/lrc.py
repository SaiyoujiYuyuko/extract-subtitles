# -*- coding: utf-8 -*-

from typing import Tuple, List, Iterator
from re import findall

MIN_MS = 60*1000
SEC_MS = 1000

def time_just(v: float, n = 2, pad = '0') -> str:
  text = str(int(v))
  return text.rjust(n, pad)

def millis2MinSecMs(ms) -> Tuple[int, int, int]:
  mins, r = divmod(ms, MIN_MS)
  secs, r = divmod(r, SEC_MS)
  return (mins, secs, r)

def millis2Time(ms) -> str:
  mins, secs, r = millis2MinSecMs(ms)
  return f"{time_just(mins)}:{time_just(secs)}.{int(r)}"

def makeConvertorFps2Ms(fps):
  ''' creates a frame-no to millis convertor '''
  return lambda no: (float(no) / fps) * SEC_MS * 100

def assembleLrc(ms, text) -> str:
  return f"[{millis2Time(ms)}] {text}"

def disassembleLrc(text) -> Tuple[int, str]:
  mm, ss, rrr, lyric = findall(r"^\[(\d{2}):(\d{2}).(\d+)\] ?(.+)$", text)[0]
  mins, secs, r = map(int, (mm, ss, rrr) )
  return (mins*MIN_MS + secs*SEC_MS + r, lyric)

def disassembleLrcFile(text, lines = lambda s: s.split("\n")) -> Iterator[Tuple[int, str]]:
  for line in lines(text):
    yield disassembleLrc(line)
