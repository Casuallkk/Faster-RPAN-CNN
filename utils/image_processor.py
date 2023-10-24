"""图片处理的一些基本操作"""
import numpy as np
from PIL import Image
import cv2


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

    @staticmethod
    def normalize(image):
        image /= 255.0
        return image

    @staticmethod
    def preprocess(image):
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = ImageProcessor.get_new_img_size(image_shape[0], image_shape[1])
        image = ImageProcessor.img2rgb(image)
        image_data = ImageProcessor.resize(image, [input_shape[1], input_shape[0]])
        return image_data

    @staticmethod
    def get_new_img_size(height, width, img_min_side=600):
        if width <= height:
            f = float(img_min_side) / width
            resized_height = int(f * height)
            resized_width = int(img_min_side)
        else:
            f = float(img_min_side) / height
            resized_width = int(f * width)
            resized_height = int(img_min_side)

        return resized_height, resized_width

    @staticmethod
    def rgb2hsv(image, hue=.1, sat=0.7, val=0.4):
        image_data = np.array(image, np.uint8)
        #   对图像进行色域变换
        #   计算色域变换的参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #   将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        #   应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
