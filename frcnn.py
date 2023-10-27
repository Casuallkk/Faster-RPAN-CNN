import colorsys
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.dataloader import FRCNNDataset
from utils.image_processor import ImageProcessor
from utils.utils_bbox import DecodeBox


class FRCNN(object):
    _defaults = {
        "model_path": 'model_data/voc_weights_vgg.pth',
        "classes_path": 'model_data/voc_classes.txt',
        "backbone": "vgg",
        "input_shape": [600, 600],
        "confidence": 0.5,
        "nms_iou": 0.3,
        'anchors_size': [8, 16, 32],
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        #   获得种类和先验框的数量
        self.class_names, self.num_classes = FRCNNDataset.get_classes(self.classes_path)

        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        #   画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):
        self.net = FasterRCNN(self.num_classes, "predict", anchor_scales=self.anchors_size, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = ImageProcessor.get_new_img_size(image_shape[0], image_shape[1])
        image_data = ImageProcessor.preprocess(image, input_shape)
        """
        image_data = np.expand_dims(np.transpose(
            ImageProcessor.normalize(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)"""

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            #   roi_cls_locs  建议框的调整参数
            #   roi_scores    建议框的种类得分
            #   rois          建议框的坐标
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                                             nms_iou=self.nms_iou, confidence=self.confidence)
            #   如果没有检测出物体，返回原图
            if len(results[0]) <= 0:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        #   图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = ImageProcessor.get_new_img_size(image_shape[0], image_shape[1])
        """image = ImageProcessor.img2rgb(image)
        image_data = ImageProcessor.resize(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(np.transpose(
            ImageProcessor.normalize(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)"""
        image_data = ImageProcessor.preprocess(image, input_shape)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                                             nms_iou=self.nms_iou, confidence=self.confidence)
            #   如果没有检测到物体，则返回原图
            if len(results[0]) <= 0:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
