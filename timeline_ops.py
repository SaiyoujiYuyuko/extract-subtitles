#!/bin/env python3
# -*- coding: utf-8 -*-

from sys import argv, stdin, stderr
from re import findall
from json import dumps, loads

from libs.fun_utils import zipWithNext
from libs.cv_utils import stringSimilarity
from libs.lrc import makeConvertorFps2Ms, assembleLrc

class Record:
  ''' Value record on the time line '''
  def __init__(self, start: int, end: int, value):
    self.start, self.end, self.value = start, end, value
  def __str__(self):
    return f"{self.start}-{self.end} {dumps(self.value, ensure_ascii=False)}"
  @staticmethod
  def loads(text):
    start, end, text = findall(r"^(\d+)-(\d+) (.*)$", text)[0]
    return Record(int(start), int(end), loads(text))

def loadsTimeline(line):
  time, text = findall(r"^(\d+) (.*)$", line)[0]
  return [int(time), loads(text)]
def dumpsTimeline(time, text):
  return f"{time} {dumps(text, ensure_ascii=False)}"

#^ Two data&representations: Record(+end) and timeline(start, text)

def openTimeline(path):
  return [loadsTimeline(ln) for ln in open(path, "r").readlines()]

def mergeDebug(path):
  for (a, b) in zipWithNext(openTimeline(path)):
    ta, sa = a; tb, sb = b
    v = stringSimilarity(sa, sb)
    print(f"{ta}-{tb} {v} {sa} | {sb}")

def merge(path, strsim_bound_max):
  bound_max = float(strsim_bound_max)
  last_text = ""
  start, end = 0, 0
  for (time, text) in openTimeline(path):
    if stringSimilarity(last_text, text) < bound_max:
      last_text = text
      end = time #< renew end
    else:
      last_text, start, end = text, time, time
      print(Record(start, end, text))

lines = lambda s: iter(s.readline, "")

def stdinSimplify():
  for line in lines(stdin):
    rec = Record.loads(line)
    print(dumpsTimeline(rec.start, rec.value))

def stdinToLRC(fps):
  ms = makeConvertorFps2Ms(float(fps))
  for line in lines(stdin):
    start, text = loadsTimeline(line)
    lrc = assembleLrc(ms(start), text)
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
