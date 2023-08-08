"""不同任务的可视化功能
"""
from typing import Any, Dict, Optional, Sequence, Tuple, Union,OrderedDict
import numpy as np
import cv2
import torch
class VisualTools():
    def classify(self, origin_mat: cv2.Mat, cur_mat: cv2.Mat, output: list):
        print("class_score:", output[0])
        print("class_index:", output[1])
    def detection(self, origin_mat: cv2.Mat, cur_mat: cv2.Mat, output: Sequence[np.array]):
        #循环所有检测的框输出并可视化
        for i, det in enumerate(output):#batch层级循环
            if len(det):
                det[:, :4] = self.scale_boxes(cur_mat.shape[2:], det[:, :4], origin_mat.shape).round()
                # Print results
                for *xyxy, conf, cls in reversed(det):
                    print(*xyxy)
                    print(conf)
                    print(cls) 
                
    def segmentation(self, origin_mat: str):
        pass

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def clip_boxes(self,boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
