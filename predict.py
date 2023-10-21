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
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    mode = "predict"
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    crop = False
    count = False
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

