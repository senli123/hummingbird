import importlib
from typing import Dict, Optional, Sequence

import ncnn
import numpy as np
import torch

from mmdeploy.utils import Backend, get_root_logger
from mmdeploy.utils.timer import TimeCounter
from ..base import BACKEND_WRAPPER, BaseWrapper

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
        