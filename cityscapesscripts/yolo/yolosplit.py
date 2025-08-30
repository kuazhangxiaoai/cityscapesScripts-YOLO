import json
import os
import cv2

from cityscapesscripts.yolo.yoloutils import GetFileFromThisRootDir, get_main_name

class split:
    def __init__(self, image_dir, anotation_dir, save_dir, img_size=1024, overlap=512, ratio=[1.0]):
        self.img_dir = image_dir
        self.anotation_dir = anotation_dir
        self.save_dir = save_dir
        self.img_size = img_size
        self.overlap = overlap
        self.ratio = ratio

        if not os.path.exists(os.path.join(self.save_dir, 'image')):
            os.mkdir(os.path.join(self.save_dir, 'image'))
        if not os.path.exists(os.path.join(self.save_dir, 'annotation')):
            os.mkdir(os.path.join(self.save_dir, 'annotation'))

    def file_name_filter(self, filenames: [], keywords: str):
        return [filename for filename in filenames if keywords in filename]

    def split_single(self, imagepath, anotationpath, ratio):
        img = cv2.imread(imagepath)
        img = cv2.resize(img, (img.shape[0] * ratio, img.shape[1] * ratio)) if ratio != 1.0 else img
        name = get_main_name(imagepath)
        anotation_name = get_main_name(anotationpath)
        anotation = cv2.imread(anotationpath)
        left, up = 0, 0
        width, height = img.shape[1], img.shape[0]
        loginfos = []
        slide = self.img_size - self.overlap
        while(left < width):
            if left + self.img_size > width:
                left = max(0, width - self.img_size)
            up = 0
            while(up < height):
                if up + self.img_size > height:
                    up = max(0, height - self.img_size)
                right = min(left + self.img_size, width)
                down = min(up + self.img_size, height)
                splitname_img = os.path.join(self.save_dir, 'image', f"{name}__{left}__{up}__{int(ratio * 10)}.png")
                splitname_msk = os.path.join(self.save_dir, 'annotation', f"{anotation_name}__{left}__{up}__{int(ratio * 10)}.png")
                patch_img = img[up: down, left: right,:]
                patch_msk = anotation[up: down, left: right, :]
                cv2.imwrite(splitname_img, patch_img)
                cv2.imwrite(splitname_msk, patch_msk)

                loginfos.append({
                    'imagename': splitname_img,
                    'annotation': splitname_msk,
                    'oriname': imagepath,
                    'oriannotation': anotationpath,
                    'left': left,
                    'up': up,
                    'right': right,
                    'down': down,
                    'resized': [img.shape[1], img.shape[0]]  #原图缩放后的尺寸
                })
                if(up + self.img_size >= height):
                    break
                else:
                    up = up + slide
            if (left + self.img_size >= width):
                break
            else:
                left = left + slide
        return loginfos

    def split_test_single(self, imagepath, ratio):
        img = cv2.imread(imagepath)
        img = cv2.resize(img, (img.shape[0] * ratio, img.shape[1] * ratio)) if ratio != 1.0 else img
        name = get_main_name(imagepath)
        left, up = 0, 0
        width, height = img.shape[1], img.shape[0]
        loginfos = []
        slide = self.img_size - self.overlap
        while (left < width):
            if left + self.img_size > width:
                left = max(0, width - self.img_size)
            up = 0
            while (up < height):
                if up + self.img_size > height:
                    up = max(0, height - self.img_size)
                right = min(left + self.img_size, width)
                down = min(up + self.img_size, height)
                splitname_img = os.path.join(self.save_dir, 'image', f"{name}__{left}__{up}__{int(ratio * 10)}.png")
                patch_img = img[up: down, left: right, :]
                cv2.imwrite(splitname_img, patch_img)

                loginfos.append({
                    'imagename': splitname_img,
                    'oriname': imagepath,
                    'left': left,
                    'up': up,
                    'right': right,
                    'down': down,
                    'resized': [img.shape[1], img.shape[0]]  # 原图缩放后的尺寸
                })
                if (up + self.img_size >= height):
                    break
                else:
                    up = up + slide
            if (left + self.img_size >= width):
                break
            else:
                left = left + slide
        return loginfos

    def run(self):
        images = GetFileFromThisRootDir(self.img_dir)
        anotats = GetFileFromThisRootDir(self.anotation_dir)
        anotats = self.file_name_filter(anotats, '_color')
        loginfos = []
        for imagepath, maskpath in zip(images, anotats):
            for r in self.ratio:
                log = self.split_single(imagepath, maskpath, r)
                loginfos = loginfos + log
                print(imagepath + '\t' + str(r) + ' : OK' )
        with open(os.path.join(self.save_dir, 'split'), "w", encoding="utf-8") as f:
            json.dump(loginfos, f, ensure_ascii=False, indent=4)
        return

    def run_test(self):
        images = GetFileFromThisRootDir(self.img_dir)
        loginfos = []
        for imagepath in images:
            for r in self.ratio:
                log = self.split_test_single(imagepath, r)
                loginfos = loginfos + log
                print(imagepath + '\t' + str(r) + ' : OK')
        with open(os.path.join(self.save_dir, 'split.json'), "w", encoding="utf-8") as f:
            json.dump(loginfos, f, ensure_ascii=False, indent=4)
        return

if __name__ == '__main__':
    splitor = split(
        '/media/yanggang/847C02507C023D84/CityEscape/leftImg8bit_trainvaltest/leftImg8bit/train',
        '/media/yanggang/847C02507C023D84/CityEscape/gtFine_trainvaltest/gtFine/train',
        save_dir='/media/yanggang/847C02507C023D84/CityEscape-YOLO/train'
    )

    splitor.run()

    splitor = split(
        '/media/yanggang/847C02507C023D84/CityEscape/leftImg8bit_trainvaltest/leftImg8bit/val',
        '/media/yanggang/847C02507C023D84/CityEscape/gtFine_trainvaltest/gtFine/val',
        save_dir='/media/yanggang/847C02507C023D84/CityEscape-YOLO/val'
    )

    splitor.run()

    splitor = split(
        '/media/yanggang/847C02507C023D84/CityEscape/leftImg8bit_trainvaltest/leftImg8bit/test',
        '/media/yanggang/847C02507C023D84/CityEscape/gtFine_trainvaltest/gtFine/test',
        save_dir='/media/yanggang/847C02507C023D84/CityEscape-YOLO/test'
    )

    splitor.run_test()