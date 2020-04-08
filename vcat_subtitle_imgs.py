#!/bin/env python3
# -*- coding: utf-8 -*-

from typing import NewType, Tuple, Dict
from PIL.Image import Image
from PIL import Image as Pillow

LTRD = NewType("LTRD", Tuple[int,int,int,int])

def vConcateImages(path_imgs: Dict[str,Image]) -> Tuple[Image, Dict[str,LTRD]]:
  width = max(map(lambda it: it.width, path_imgs.values()))
  height = sum(map(lambda it: it.height, path_imgs.values()))
  newImage = Pillow.new("RGBA", (width, height))
  iter_img = iter(path_imgs.items())
  areas = {}
  y = 0
  while y < height:
    (path, img) = next(iter_img)
    box = (0,y, img.width,y+img.height)
    areas[path] = box
    newImage.paste(img, box)
    y += img.height
  return (newImage, areas)

def saveCropImages(img: Image, areas: Dict[str,LTRD]):
  for (k, v) in areas.items():
    img.crop(v).save(k)

from re import findall
from json import dump, load
def main(args):
  if len(args) == 0:
    print("Usage: dst src... | unpack dst")
    return
  if args[0] == "unpack":
    path = args[1]
    areas = load(open(f"{path}.json", "r"))
    saveCropImages(Pillow.open(path), areas)
  else:
    dst = args[0]
    srcs = sorted(args[1:], key=lambda s: int(findall(r"(\d+)", s)[0]))
    print(f"{dst} << {srcs[:5]}..{srcs[-5:]} ({len(srcs)})")

    (img, areas) = vConcateImages({path: Pillow.open(path) for path in srcs})
    img.save(dst); dump(areas, open(f"{dst}.json", "w+"))

from sys import argv
if __name__ == "__main__": main(argv[1:])
