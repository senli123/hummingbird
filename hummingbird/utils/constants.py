from enum import Enum

class AdvancedEnum(Enum):
    """定义枚举类"""
    @classmethod
    def get(cls, value):
        for k in cls:
            if k.value == value:
                return k
        raise KeyError(f'Cannot get key by value "{value}" for {cls}')


#定义目前使用的backend
class Backend(AdvancedEnum):
    PYTORCH = 'pytorch'
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    PPLNN = 'pplnn'
    NCNN = 'ncnn'
    SNPE = 'snpe'
    OPENVINO = 'openvino'
    SDK = 'sdk'
    TORCHSCRIPT = 'torchscript'
    RKNN = 'rknn'
    ASCEND = 'ascend'
    COREML = 'coreml'
    DEFAULT = 'default'