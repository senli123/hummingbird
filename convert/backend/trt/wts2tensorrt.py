from .model import * 
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
class ModelFactory():
    def build_model(self,name):
        return eval(name)()
    
def wts2tensorrt(   name: str, 
                    wts_path: str,
                    input_blob_name: str,
                    output_blob_name:str,
                    max_batch_size: int,
                    input_h: int,
                    input_w: int,
                    output_file_prefix:str):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    model_factory = ModelFactory()
    register_model = model_factory.build_model(name)
    engine = register_model.create_engine(builder,
                    config, 
                    wts_path,
                    trt.float32,
                    input_blob_name,
                    output_blob_name,
                    max_batch_size, 
                    input_h,
                    input_w)
    assert engine
    with open(output_file_prefix + '.engine', "wb") as f:
        f.write(engine.serialize())

    del engine
    del builder
    del config