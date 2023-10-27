import time
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    origin_path = "img/"
    save_path = "img_out/"

    print("Start prediction...")
    img_names = os.listdir(origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith('.jpg'):
            image_path = os.path.join(origin_path, img_name)
            image = Image.open(image_path)
            result_image = frcnn.detect_image(image)
            plt.imshow(result_image)
            plt.show()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            result_image.save(os.path.join(
                save_path, img_name), quality=95, subsampling=0)

