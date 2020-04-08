#!/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image

def vConcateImages(imgs) -> Image:
  width = max(map(lambda it: it.width, imgs))
  height = sum(map(lambda it: it.height, imgs))
  newImage = Image.new("RGBA", (width, height))
  iter_img = iter(imgs)
  y = 0
  while y < height:
    img = next(iter_img)
    newImage.paste(img, box=(0,y, img.width, y+img.height))
    y += img.height
  return newImage

from re import findall
def main(args):
  dst = args[0]
  srcs = sorted(args[1:], key=lambda s: int(findall(r"(\d+)", s)[0]))
  print(f"{dst} << {srcs[:5]}..{srcs[-5:]} ({len(srcs)})")
  vConcateImages([Image.open(path) for path in srcs]).save(dst)

from sys import argv
if __name__ == "__main__": main(argv[1:])
