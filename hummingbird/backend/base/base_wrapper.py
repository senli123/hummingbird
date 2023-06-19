from abc import ABCMeta, abstractclassmethod
from typing import Dict, List, Sequence

import torch

class BaseWrapper(torch.nn.Module, metaclass = ABCMeta):

    def __init__(self, output_names: Sequence[str]):
        super()._init__()
        self._output_names = output_names

    @staticmethod
    def get_backend_file_count() -> int:

        return 1
    
    @abstractclassmethod
    def forward(self, input: Dict[str,
                                    torch.Tensor]) ->Dict[str,torch.Tensor]:
        pass

    @property
    def output_names(self):
        return self._output_names
    
    @output_names.setter
    def output_names(self,value):
        self._output_names = value

    
    def output_to_list(self, output_dict: Dict[str, torch.Tensor]) -> \
        List[torch.Tensor]:

        outputs = [output_dict[name] for name in self._output_names]
        return outputs
