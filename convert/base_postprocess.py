from abc import ABCMeta,abstractclassmethod
import cv2
import time
import numpy as np
import torch,torchvision
import logging


class PostprocessFactory():
    def create_postprocess(self,name,kwargs):
        return eval(name)(**kwargs) 
    
class BasePostprocess(metaclass = ABCMeta):
    
    @abstractclassmethod
    def run(self, mat):
        pass
    @abstractclassmethod
    def get_name(self):
        pass

class ClassifyWrapper(BasePostprocess):
    def __init__(self,class_nums,top_nums):
        self.class_nums = class_nums
        self.top_nums = top_nums
    def run(self, output):
        output_index = np.argmax(output)
        score = output[output_index]
        out = [score, output_index]
        return out 
    def get_name(self,):
        return "ClassifyWrapper"
    

class Yolov5PostprocessWrapper(BasePostprocess):
    def __init__(self,class_nums,conf_thres, iou_thres):
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        self.class_nums = class_nums
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
    def run(self, prediction):
    #     bs = prediction.shape[0]  # batch size
    #     nc = self.class_nums
    #     #第一次筛选，将框的score大于conf_thres的候选框提取出来
    #     xc = prediction[..., 4] > self.conf_thres  # candidates 

    #     # Settings
    #     # min_wh = 2  # (pixels) minimum box width and height
    #     max_wh = 7680  # (pixels) maximum box width and height
    #     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    #     time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    #     redundant = True  # require redundant detections
    #     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    #     merge = False  # use merge-NMS

    #     # t = time.time()
    #     mi = 5 + nc  # mask start index
    #     output = [np.zeros((0, 6 + self.class_nums))] * bs
    #     #循环所有batch
    #     for xi, x in enumerate(prediction):  # image index, image inference
    #         # Apply constraints
    #         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
    #         #把第一次筛选的所有框拿出来
    #         x = x[xc[xi]]  # confidence

            

    #         # If none remain process next image
    #         if not x.shape[0]:
    #             continue

    #         # Compute conf
    #         #类别得分与框的得分相乘，为第二次筛选做准备
    #         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    #         # Box/Mask
    #         #将所有候选框的坐标进行转换
    #         box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
    #         mask = x[:, mi:]  # zero columns if no masks
    #         #拿出候选框的最大类别的得分和对应的类别index
    #         conf, j = np.max(x[:, 5:mi],axis = 1, keepdims = True),np.argmax(x[:,5:mi],axis=1)
    #         x = np.concatenate((box, conf, j.float(), mask), 1)[conf.view(-1) > self.conf_thres]
    #         # Check shape
    #         n = x.shape[0]  # number of boxes
    #         if not n:  # no boxes
    #             continue
    #         #按socre排序
    #         x = x[np.argsort(-x[:,4])][:max_nms]
           
    #         # Batched NMS
    #         c = x[:, 5:6] *  max_wh # classes
    #         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    #         i = torchvision.ops.nms(boxes, scores, self.iou_thres)  # NMS
    #         i = i[:max_det]  # limit detections
    #         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
    #             # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
    #             iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
    #             weights = iou * scores[None]  # box weights
    #             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
    #             if redundant:
    #                 i = i[iou.sum(1) > 1]  # require redundancy

    #         output[xi] = x[i]
    #         if mps:
    #             output[xi] = output[xi].to(device)
    #         if (time.time() - t) > time_limit:
    #             LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
    #             break  # time limit exceeded

    #     return output

    # def xywh2xyxy(self,x):
    #     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #     y = np.copy(x)
    #     y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    #     y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    #     y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    #     y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    #     return y
        labels = ()
        nm = 0
        agnostic = False
        classes = None
        max_det=300
        multi_label = False
        prediction = prediction.reshape(1,25200,85)
        prediction = torch.from_numpy(prediction)
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
       
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > self.conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > self.conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > self.conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > self.iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded
        #返回前将torch转换成numpy
        np_output = []
        for i in range(len(output)):
            torch_out = output[i]
            np_output.append(torch_out.numpy())
        return np_output
    
    def xywh2xyxy(self,x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    def box_iou(self,box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def get_name(self,):
        return "ClassifyWrapper"
    
    