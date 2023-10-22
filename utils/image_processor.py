"""图片处理的一些基本操作"""
import numpy as np
import cv2
from PIL import Image


class ImageProcessor:
    def __init__(self, img):
        self.img = img

    @staticmethod
    def img2rgb(image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    @staticmethod
    def resize(image, size):
        w, h = size
        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image
