"""onnx转ncnn ir
"""
import os
import os.path as osp
import tempfile
from subprocess import call
from typing import List, Optional, Union

import onnx

from .init_plugins import get_onnx2ncnn_path

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get_output_model_file(onnx_path: str,
                          work_dir: Optional[str] = None) -> List[str]:
    """Returns the path to the .param, .bin file with export result.

    Args:
        onnx_path (str): The path to the onnx model.
        work_dir (str|None): The path to the directory for saving the results.
            Defaults to `None`, which means use the directory of onnx_path.

    Returns:
        List[str]: The path to the files where the export result will be
            located.
    """
    if work_dir is None:
        work_dir = osp.dirname(onnx_path)
    mkdir_or_exist(osp.abspath(work_dir))
    file_name = osp.splitext(osp.split(onnx_path)[1])[0]
    save_param = osp.join(work_dir, file_name + '.param')
    save_bin = osp.join(work_dir, file_name + '.bin')
    return [save_param, save_bin]

def from_onnx(onnx_model:Union[onnx.ModelProto, str],
                output_file_prefix: str):
    if not isinstance(onnx_model, str):
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)
    else:
        onnx_path = onnx_model

    save_param = output_file_prefix + '.param'
    save_bin = output_file_prefix + '.bin'

    onnx2ncnn_path = get_onnx2ncnn_path()
    ret_code = call([onnx2ncnn_path, onnx_path, save_param, save_bin])
    assert ret_code == 0, 'onnx2ncnn failed'