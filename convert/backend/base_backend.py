from abc import ABCMeta,abstractclassmethod
from trt import TensorRTWrapper
class BaseBackend(metaclass = ABCMeta):
    
    @abstractclassmethod
    def infer(self, input):
        pass
    @abstractclassmethod
    def get_name(self):
        pass

class BackendFactory():
    def create_backend(self,name):
        return eval(name)()