from doctest import OutputChecker
import importlib
from typing import Dict, Optional, Sequence

import ncnn
import numpy as np
import torch

from hummingbird.utils import Backend, get_root_logger
from hummingbird.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper

@BACKEND_WRAPPER.register_module(Backend.NCNN.value)
class NCNNWrapper(BaseWrapper):
    def __init__(self, 
                 param_file: str,
                 bin_file: str,
                 output_names: Optional[Sequence[str]] = None,
                 use_vulkan: bool = False,
                 **kwargs):
        #初始化模型
        net = ncnn.Net()
        #加载ncnn扩充的op
        if importlib.util.find_spec('hummingbird.backend.ncnn.ncnn_ext'):
            from hummingbird.backend.ncnn import ncnn_ext
            ncnn_ext.register_mmdeploy_custom_layers(net)
        net.opt.use_vulkan_compute = use_vulkan
        net.load_param(param_file)
        net.load_model(bin_file)

        self._net = net

        if output_names is None:
            assert hasattr(self._net, 'output_names')
            output_names = self._net.output_names
        
        super().__init__(output_names)

    @staticmethod
    def get_backend_file_count() -> int:
        return 2
    
    def forward(self, inputs: Dict[str, torch.Tensor]) ->Dict[str, torch.Tensor]:

        input_list = list(inputs.values())
        batch_size = input_list[0].size(0)
        logger = get_root_logger()
        if batch_size > 1:
            logger.warning(
                f'ncnn only support batch_size = 1, but given {batch_size}'
            )

        for input_tensor in input_list[1:]: 
            assert input_tensor.size(0) == batch_size, \
                "All tensors should have same batch size"
            assert input_tensor.device.type == 'cpu', \
            'ncnn only supports cpu device'

        output_names = self._output_names

        outputs = dict([name , [None]* batch_size] for name in output_names)

        for batch_id in range(batch_size):
            ex = self._net.create_extractor()
            #set inputs
            for name, input_tensor in inputs.items():
                data = input_tensor[batch_id].contiguous()
                data = data.detach().cpu().numpy()
                input_mat = ncnn.Mat(data)
                input_mat = input_mat.clone()
                ex.input(name, input_mat)
        
            #get outputs
            result = self.__ncnn_execute(extractor = ex, output_names = output_names)

            for name in output_names:
                mat = result[name]
                if mat.empty():
                    logger.warning(
                        f'The "{name}" output of ncnn model is empty.'
                    )
                    continue
                outputs[name][batch_id] = torch.from_numpy(np.array(mat))

        #将一个batch的output concat起来
        for name, output_tensor in outputs.items():
            if None in output_tensor:
                outputs[name] = None
            else:
                outputs[name] = torch.stack(output_tensor)
        
        return outputs

    
    @TimeCounter.count_time(Backend.NCNN.value)
    def __ncnn_execute(self, extractor: ncnn.Extractor,
                       output_names: Sequence[str]) -> Dict[str, ncnn.Mat]:
        result = {}
        for name in output_names:
            out_ret, out = extractor.extract(name)
            assert out_ret == 0, f'Failed to extract output: {out}.'
            result[name] = out
        return result