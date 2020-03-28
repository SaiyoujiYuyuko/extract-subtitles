#!/bin/env python3
# -*- coding: utf-8 -*-

from sys import argv, stdin, stderr
from re import findall
from json import dumps, loads

from libs.fun_utils import zipWithNext
from libs.cv_utils import stringSimilarity
from libs.lrc import makeConvertorFps2Ms, dumpsLrc

class Record:
  ''' Value record on the time line '''
  def __init__(self, start: int, end: int, value):
    self.start, self.end, self.value = start, end, value
  def __str__(self):
    return f"{self.start}-{self.end} {dumps(self.value, ensure_ascii=False)}"
  @staticmethod
  def loads(line):
    start, end, text = findall(r"^(\d+)-(\d+) (.*)$", line)[0]
    return Record(int(start), int(end), loads(text))

class Timeline:
  def __init__(self, time: int, value):
    self.time, self.value = time, value
  def __str__(self):
    return f"{self.time} {dumps(self.value, ensure_ascii=False)}"
  @staticmethod
  def loads(line):
    time, text = findall(r"^(\d+) (.*)$", line)[0]
    return [int(time), loads(text)]

#^ Two data&representations: Record(+end) and timeline(time, text)

def openTimeline(path):
  return [Timeline.loads(ln) for ln in open(path, "r").readlines()]

def mergeDebug(path):
  for (a, b) in zipWithNext(openTimeline(path)):
    ta, sa = a; tb, sb = b
    v = stringSimilarity(sa, sb)
    print(f"{ta}-{tb} {str(v)[0:4].ljust(4, '0')} {sa} | {sb}")

def merge(path, strsim_bound_max):
  bound_max = float(strsim_bound_max)
  last_text = ""
  start, end = 0, 0
  for (time, text) in openTimeline(path):
    if stringSimilarity(last_text, text) < bound_max:
      last_text = text
      end = time #< renew end
    else:
      print(Record(start, end, last_text))
      last_text, start, end = text, time, time

lines = lambda s: iter(s.readline, "")

def stdinSimplify():
  for line in lines(stdin):
    rec = Record.loads(line)
    print(Timeline(rec.start, rec.value))

def stdinToLRC(fps):
  ms = makeConvertorFps2Ms(float(fps))
  for line in lines(stdin):
    start, text = Timeline.loads(line)
    lrc = dumpsLrc(ms(start), text)
    print(lrc)

handler = { "merge-debug": mergeDebug, "merge": merge, "simplify": stdinSimplify, "to-lrc": stdinToLRC }

def main(args):
  if len(args) == 0:
    tl = "timeline_file"
    print(f"Usage: merge-debug <{tl}> | merge <{tl}> <strsim_bound_max> | simplify | to-lrc <fps>", file=stderr)
    return
  key_op = args[0]
  handler[key_op](*args[1:])

if __name__ == "__main__": main(argv[1:])
