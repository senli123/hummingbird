from abc import ABCMeta,abstractclassmethod
import cv2
import numpy as np

class PreprocessFactory():
    def create_preprocess(self,name,**kwargs):
        return eval(name)(kwargs)
    
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
    def run(self, mat):
        if self.bgr2rgb:
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        mat = cv2.resize(mat,self.size)
        return mat
    def get_name(self,):
        return "ResizeWrapper"
    
class NormalizeWrapper(BasePreprocess):
    def __init__(self,mean,std):
        self.mean = np.array(mean, dtype= np.float32) * 255
        self.size = np.reciprocal(np.array(std, dtype= np.float32)* 255, dtype=np.float32)
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