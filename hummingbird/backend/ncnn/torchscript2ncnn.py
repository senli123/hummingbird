"""torchscript转ncnn ir
"""
import os
import os.path as osp
import tempfile
from subprocess import call
from typing import List, Optional, Union
import torch
import onnx

from .init_plugins import get_torchscript2ncnn_path

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get_output_model_file(pt_path: str,
                          work_dir: Optional[str] = None) -> List[str]:
    """Returns the path to the .param, .bin file with export result.

    Args:
        pt_path (str): The path to the pt model.
        work_dir (str|None): The path to the directory for saving the results.
            Defaults to `None`, which means use the directory of pt_path.

    Returns:
        List[str]: The path to the files where the export result will be
            located.
    """
    if work_dir is None:
        work_dir = osp.dirname(pt_path)
    mkdir_or_exist(osp.abspath(work_dir))
    file_name = osp.splitext(osp.split(pt_path)[1])[0]
    save_param = osp.join(work_dir, file_name + '.param')
    save_bin = osp.join(work_dir, file_name + '.bin')
    return [save_param, save_bin]

def from_torchscrip(torchscript_model:Union[torch.nn.Module, str],
                output_file_prefix: str):
    if not isinstance(torchscript_model, str):
        pt_path = tempfile.NamedTemporaryFile(suffix='.pt').name
        onnx.save(torchscript_model, pt_path)
    else:
        pt_path = torchscript_model

    save_param = output_file_prefix + '.param'
    save_bin = output_file_prefix + '.bin'

    pnnx_path = get_torchscript2ncnn_path()
    ret_code = call([pnnx_path, pt_path, save_param, save_bin])
    assert ret_code == 0, 'torchscript2ncnn failed'