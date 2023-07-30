import os
import ctypes
import tensorrt as trt
from packaging import version
import onnx
from typing import Dict, Optional, Sequence, Union
def load_tensorrt_plugin() ->bool:
    tensorrt_plugin_path = '../../lib/libmmdeploy_tensorrt_ops.so'
    if os.path.exists(tensorrt_plugin_path):
        ctypes.CDLL(tensorrt_plugin_path)
        print('Successfully loaded tensorrt plugins from {0}'.format(tensorrt_plugin_path))
    else:
        print('Could not load the library of tensorrt plugins. \
            Because the file does not exist: {0}'.format(tensorrt_plugin_path))


def onnx2tensorrt(  onnx_path: str,
                    output_file_prefix: str,
                    input_shapes: Dict[str, Sequence[int]],
                    max_workspace_size, int = 0,
                    device_id: int = 0,
                    fp16_mode: bool = False,
                    int8_mode: bool = False,
                    int8_param: Optional[dict] = None,
                  ):
    os.environ['CUDA_DEVICE'] = str(device_id)
    #加载plugin算子
    load_tensorrt_plugin()
    logger = trt.Logger()
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    onnx_model = onnx.load(onnx_path)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # config builder
    if version.parse(trt.__version__) < version.parse('8'):
        builder.max_workspace_size = max_workspace_size

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    # cuda_version = search_cuda_version()
    # if cuda_version is not None:
    #     version_major = int(cuda_version.split('.')[0])
    #     if version_major < 11:
    #         # cu11 support cublasLt, so cudnn heuristic tactic should disable CUBLAS_LT # noqa E501
    #         tactic_source = config.get_tactic_sources() - (
    #             1 << int(trt.TacticSource.CUBLAS_LT))
    #         config.set_tactic_sources(tactic_source)

    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        if version.parse(trt.__version__) < version.parse('8'):
            builder.fp16_mode = fp16_mode
        config.set_flag(trt.BuilderFlag.FP16)

    if int8_mode:
        from .calib_utils import HDF5Calibrator
        config.set_flag(trt.BuilderFlag.INT8)
        assert int8_param is not None
        config.int8_calibrator = HDF5Calibrator(
            int8_param['calib_file'],
            input_shapes,
            model_type=int8_param['model_type'],
            device_id=device_id,
            algorithm=int8_param.get(
                'algorithm', trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2))
        if version.parse(trt.__version__) < version.parse('8'):
            builder.int8_mode = int8_mode
            builder.int8_calibrator = config.int8_calibrator

    # create engine
    engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    with open(output_file_prefix + '.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
    return engine
    


