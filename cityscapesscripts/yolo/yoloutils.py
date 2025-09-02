import json
import os
import cv2

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def get_main_name(filename: str):
    return filename.split(os.sep)[-1].split('.')[0]

def get_image_name(imagename):
  main_name = get_main_name(imagename)
  return main_name.replace('leftImg8bit', '') #''/media/yanggang/847C02507C023D84/CityEscape/gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_.png''