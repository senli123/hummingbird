from abc import ABCMeta,abstractclassmethod

class BaseBackend(metaclass = ABCMeta):
    
    @abstractclassmethod
    def infer(self, input):
        pass
    @abstractclassmethod
    def get_name(self):
        pass

