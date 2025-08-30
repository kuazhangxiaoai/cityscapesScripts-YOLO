import json
import os
import cv2
import numpy as np

from cityscapesscripts.yolo.yoloutils import GetFileFromThisRootDir, get_main_name

class merge:
    def __init__(self, img_split_dir, annotation_split_dir, img_dir, splitfile, save_dir):
        self.img_split_dir = img_split_dir
        self.annotation_split_dir = annotation_split_dir
        self.split_file = splitfile
        self.img_dir = img_dir
        self.save_dir = save_dir
        if not os.path.exists(os.path.join(self.save_dir, 'image')):
            os.mkdir(os.path.join(self.save_dir, 'image'))
        if not os.path.exists(os.path.join(self.save_dir, 'annotation')):
            os.mkdir(os.path.join(self.save_dir, 'annotation'))

    def run(self):
        ori_images= GetFileFromThisRootDir(self.img_dir)

        with open(self.split_file, 'r') as f:
            split_info = json.load(f)

        for imagename in ori_images:
            selects_indexes = [i for i, item in enumerate(split_info) if item.get("oriname") == imagename]
            selects = [split_info[i] for i in selects_indexes]
            h, w = cv2.imread(imagename).shape[:-1]
            s = selects[0]['resized'] #(width, height)
            merged_image = np.zeros([s[1], s[0], 3], dtype=np.uint8)
            merged_annotation = np.zeros([s[1], s[0], 3], dtype=np.uint8)
            for item in selects:
                l, r, t, b = item['left'],item['right'],item['up'],item['down']
                merged_image[t: b, l: r, :] = cv2.imread(item['imagename'])
                merged_annotation[t: b, l: r, :] = cv2.imread(item['annotation'])

            resized_image = cv2.resize(merged_image, (w, h))
            resized_annotation = cv2.resize(merged_annotation, (w, h))

            cv2.imwrite(os.path.join(self.save_dir, 'image', get_main_name(imagename)+'.png'), resized_image)
            cv2.imwrite(os.path.join(self.save_dir, 'annotation', get_main_name(imagename)+'.png'), resized_annotation)


if __name__ == '__main__':
    merger = merge(
        '/media/yanggang/847C02507C023D84/CityEscape-YOLO/val/image',
        '/media/yanggang/847C02507C023D84/CityEscape-YOLO/val/annotation',
        '/media/yanggang/847C02507C023D84/CityEscape/leftImg8bit_trainvaltest/leftImg8bit/val',
        '/media/yanggang/847C02507C023D84/CityEscape-YOLO/val/split',
        '/media/yanggang/847C02507C023D84/CityEscape-YOLO/merge'
    )
    merger.run()





