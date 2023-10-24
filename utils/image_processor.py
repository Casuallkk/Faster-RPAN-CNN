"""图片处理的一些基本操作"""
import numpy as np
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
