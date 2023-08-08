from abc import ABCMeta,abstractclassmethod
import cv2
import numpy as np

class PreprocessFactory():
    def create_preprocess(self,name,kwargs):
        return eval(name)(**kwargs)
    
class BasePreprocess(metaclass = ABCMeta):
    
    @abstractclassmethod
    def run(self, mat):
        pass
    @abstractclassmethod
    def get_name(self):
        pass


class ResizeWrapper(BasePreprocess):
    def __init__(self,bgr2rgb,size,keep_ratio):
        self.bgr2rgb = bgr2rgb
        self.size = size
        self.keep_ratio = keep_ratio
    def run(self, mat):
        if self.bgr2rgb:
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        mat = cv2.resize(mat,self.size)
        return mat
    def get_name(self,):
        return "ResizeWrapper"
    

class NormalizeWrapper(BasePreprocess):
    def __init__(self,mean,std, limit, normalize):
        self.mean_array = np.array(mean, dtype= np.float32) * 255
        self.std_array = np.reciprocal(np.array(std, dtype= np.float32)* 255, dtype=np.float32)
    def run(self, mat):
        mat = np.array(mat, dtype=np.float32, order='C')
        mat -= self.mean_array
        mat *= self.std_array
        mat = np.transpose(mat, [2,0,1])
        mat = np.expand_dims(mat, axis=0)
        mat = np.array(mat, dtype=np.float32, order='C')
        return mat

    def get_name(self,):
        return "NormalizeWrapper"
    

class Yolov5PreprocessWrapper(BasePreprocess):
    
    def __init__(self,input_h,input_w,color,stride):
        self.input_h = input_h
        self.input_w = input_w
        self.color = color
        self.stride = stride
    def run(self, mat):
        shape = mat.shape[:2] #(height, width)
        #计算scale
        r = min(self.input_h/shape[0], self.input_w / shape[1])
        #计算padding
        ratio = r,r
        new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r)) #(w,h)
        dw,dh = self.input_w - new_unpad[0], self.input_h - new_unpad[1]
        #dw,dh = np.mod(dw,self.stride), np.mod(dh, self.stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            mat = cv2.resize(mat, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        mat = cv2.copyMakeBorder(mat, top, bottom, left, right, cv2.BORDER_CONSTANT, value = self.color)
        mat = mat.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        mat = mat.astype(np.float32)
        # Normalize to [0,1]
        mat /= 255.0
        # CHW to NCHW format
        mat = np.expand_dims(mat, axis=0)
        # Convert the image to row-major order, also known as "C order":
        mat = np.ascontiguousarray(mat)
        return mat

        
        

    def get_name(self,):
        return "Yolov5PreprocessWrapper"