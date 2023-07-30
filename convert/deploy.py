"""根据配置文件完成指定模型转换和结果对比
"""
import json
import os
from const import *
import torch
from pytorch2ir import Pytorch2ir
from backend import onnx2tensorrt, wts2tensorrt, BackendFactory
from base_preprocess import PreprocessFactory
from base_postprocess import PostprocessFactory
import cv2
class Pipeline:
    def __init__(self,pth_path, preprocess_args,postprocess_args,backend_args):
        if os.path.exists(pth_path):
            self.net = torch.load(pth_path)
        else:
            self.net = None
        #注册preprocess
        preprocess_factory = PreprocessFactory()
        self.preprocess_pipeline = []
        for cur_preprocess_arg in preprocess_args:
            preprocess_type = cur_preprocess_arg['type']
            preprocess_params = cur_preprocess_arg['params']
            self.preprocess_pipeline.append(preprocess_factory.create_preprocess(preprocess_type, preprocess_params))
        #注册backend
        backend_factory = BackendFactory()
        self.backend_pipeline = []
        for cur_backend_arg in backend_args:
            backend_type = cur_backend_arg['type']
            backend_params = cur_backend_arg['params']
            self.backend_pipeline.append(backend_factory.create_backend(backend_type, backend_params))
        #注册postprocess
        postprocess_factory = PostprocessFactory()
        self.postprocess_pipeline = []
        for cur_postprocess_arg in postprocess_args:
            postprocess_type = cur_postprocess_arg['type']
            postprocess_params = cur_postprocess_arg['params']
            self.postprocess_pipeline.append(postprocess_factory.create_postprocess(postprocess_type, postprocess_params))

    def infer(self, mat:cv2.Mat):
        for preprocess_node in self.preprocess_pipeline:
            mat = preprocess_node.run(mat)
        pytorch_output = self.net(mat)
        backend_output = self.backend_pipeline[0].infer(mat)
        self.postprocess_pipeline[0].run(pytorch_output)
        self.postprocess_pipeline[0].run(backend_output)
    def destory(self,):
        self.backend_pipeline[0].destory()

def main(config_json_path):
    if not os.path.exists(config_json_path):
        print("please check your input config_json_path")
        return
    with open(config_json_path) as f:
        config_info = json.load(f)
        #分别获取转换和前后处理的信息
        convert_info = config_info['Convert']
        backend_convert_info = config_info['BackendConvert']
        preprocess_info = config_info['Preprocess']
        infer_info = config_info['Infer']
        postprocess_info = config_info['Postprocess']
        #检查针对当前这种backcend，该convert类型是否支持
        backend_type = infer_info['type']
        convert_type = convert_info['type']
        if backend_type not in BACKEND_IR_DICT.keys():
            print("we do not support this backend:{0}".format(backend_type))
            return
        if convert_type not in BACKEND_IR_DICT[backend_type]:
            print("we do not support this convert_type:{0} in {1}".format(convert_type, backend_type))
            return
        #先进行指定ir的转换
        convert_params = convert_info['params']
        pth_path = convert_params['pth_path']
        ir_path = convert_params['ir_path']
        #如果ir_path存在则不进行转换，直接进行指定backend的ir转换
        if not os.path.exists(pth_path):
            print("please check you pth_path")
            return 
        if not os.path.exists(ir_path):
            convert_ir_class = Pytorch2ir()
            convert_ir_func = getattr(convert_ir_class,convert_type)
            convert_ir_func(**convert_params)
        #进行backend_ir的转换
        backend_convert_type = backend_convert_info['type']
        backend_convert_params = backend_convert_info['params']
        if backend_convert_type == "onnx2tensorrt":
            onnx2tensorrt(backend_convert_params)
        elif backend_convert_type == "wts2tensorrt":
            wts2tensorrt(backend_convert_params)

        #转换成功之后创建推理pipeline，检查转换的结果
        pipeline = Pipeline(pth_path, preprocess_info,postprocess_info,infer_info)
        image_path = config_info["config_info"]
        mat = cv2.imread(image_path)
        pipeline.infer(mat)
        pipeline.destory()


config_json_path = "config/convert/example.json"
main(config_json_path)