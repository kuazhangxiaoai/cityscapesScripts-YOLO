import json
import os
import random
import shutil
from cityscapesscripts.yolo.yoloutils import GetFileFromThisRootDir, get_main_name

class sampler:
    def __init__(self, image_dir, mask_dir, save_dir, r=0.1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.save_dir = save_dir
        self.r = r

        if not os.path.exists(os.path.join(self.save_dir, 'image')):
            os.mkdir(os.path.join(self.save_dir, 'image'))
        if not os.path.exists(os.path.join(self.save_dir, 'annotation')):
            os.mkdir(os.path.join(self.save_dir, 'annotation'))

    def run(self):
        imagefiles = GetFileFromThisRootDir(self.image_dir)
        maskfiles = GetFileFromThisRootDir(self.mask_dir)

        for imagefile, maskfile in zip(imagefiles, maskfiles):
            if random.random() < self.r:
                name = imagefile.split(os.sep)[-1]
                shutil.copy(imagefile, os.path.join(self.save_dir, 'image', name))
                shutil.copy(maskfile, os.path.join(self.save_dir, 'annotation', name))

    def run_test(self):
        imagefiles = GetFileFromThisRootDir(self.image_dir)

        for imagefile in imagefiles:
            if random.random() < self.r:
                name = imagefile.split(os.sep)[-1]
                shutil.copy(imagefile, os.path.join(self.save_dir, 'image', name))


if __name__ == "__main__":
    s = sampler('/media/yanggang/847C02507C023D84/CityEscape-YOLO/val/image',
                '/media/yanggang/847C02507C023D84/CityEscape-YOLO/val/annotation',
                '/media/yanggang/847C02507C023D84/CityEscape-YOLO-small/val',
                0.05)
    s.run()