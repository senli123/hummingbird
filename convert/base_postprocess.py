from abc import ABCMeta,abstractclassmethod
import cv2
import numpy as np

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
        return score, output_index
    def get_name(self,):
        return "ClassifyWrapper"
    