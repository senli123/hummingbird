from .trt import TensorrtWrapper
class BackendFactory():
    def create_backend(self,name,kwargs):
        return eval(name)(**kwargs)