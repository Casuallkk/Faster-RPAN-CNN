import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from frcnn import FRCNN


class GetMap:
    """
    工具类
    注：若希望修改yolo的index以获得不同注意力机制下的数据，
    请在yolo.py中提前对model path, index, class path进行修改
    """
    def __init__(self, classes_path, score_threhold, map_out_path,
                 MINOVERLAP=0.5, confidence=0.001, nms_iou=0.5, map_vis=True):
        self.classes_path = classes_path
        self.score_threhold = score_threhold
        # 门限值
        self.MINOVERLAP = MINOVERLAP
        # MINOVERLAP用于指定想要获得的mAPx， 如AP0.5为MINOVERLAP = 0.5
        self.confidence = confidence
        # confiden为预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
        self.nms_iou = nms_iou
        self.map_vis = map_vis
        # 是否可视化效果，默认为True
        self.map_out_path = map_out_path
        self.class_names, _ = get_classes(classes_path)

        if not os.path.exists(map_out_path):
            os.makedirs(map_out_path)
        if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
            os.makedirs(os.path.join(map_out_path, 'ground-truth'))
        if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
            os.makedirs(os.path.join(map_out_path, 'detection-results'))
        if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
            os.makedirs(os.path.join(map_out_path, 'images-optional'))

    def get_predict_results(self, ids, VOCdevkit_path):
        """
        获取预测结果
        :param ids:Target image ids (list)
        :param VOCdevkit_path: Target dataset folder(str)
        :return: None
        """
        print("Load model.")
        frcnn = FRCNN(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")
        print("Get predict result.")
        for id in tqdm(ids):
            image_path = os.path.join(
                VOCdevkit_path, "VOC/JPEGImages/" + id + ".jpg")
            image = Image.open(image_path)
            if self.map_vis:
                image.save(os.path.join(
                    self.map_out_path, "images-optional/" + id + ".jpg"))
            frcnn.get_map_txt(id, image, self.class_names, self.map_out_path)
        print("Get predict result done.")

    def get_ground_truth(self, ids, VOCdevkit_path):
        """
        Get ground truth
        :param ids: Target image ids (list)
        :param VOCdevkit_path: Target dataset folder(str)
        :return: None
        """
        print("Get ground truth result.")
        for id in tqdm(ids):
            with open(os.path.join(self.map_out_path, "ground-truth/" + id + ".txt"), "w") as new_f:
                root = ET.parse(
                    os.path.join(
                        VOCdevkit_path, "VOC/Annotations/" + id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in self.class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    def get_map(self):
        """
        According to the score_threhold to get AP.
        :return: None
        """
        print("Get map.")
        get_map(self.MINOVERLAP, True, score_threhold=self.score_threhold, path=self.map_out_path)
        print("Get map done.")

    def get_Coco_map(self):
        """
        Get AP through coco toolsbox
        需要pycocotools！！！！！
        :return: None
        """
        print("Get map.")
        get_coco_map(class_names=self.class_names, path=self.map_out_path)
        print("Get map done.")


if __name__ == "__main__":
    classes_path = 'model_data/voc_classes.txt'
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    MINOVERLAP = 0.5
    confidence = 0.02
    nms_iou = 0.5
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    score_threhold = 0.5
    #   map_vis用于指定是否开启VOC_map计算的可视化
    map_vis = True
    VOCdevkit_path = 'VOCdevkit'
    map_out_path = 'map_out'

    ap = GetMap(classes_path, score_threhold, map_out_path, MINOVERLAP,
                confidence, nms_iou, map_vis)

    image_ids = open(
        os.path.join(VOCdevkit_path, "VOC/ImageSets/Main/test.txt")).read().strip().split()

    # 推荐顺序
    ap.get_predict_results(ids=image_ids, VOCdevkit_path=VOCdevkit_path)
    ap.get_ground_truth(image_ids, VOCdevkit_path)
    ap.get_map()
    ap.get_Coco_map()
