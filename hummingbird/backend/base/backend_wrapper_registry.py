from hummingbird.utils.registry import Registry
from hummingbird.utils.config_utils import Backend

#定义backend_wrapper的构建函数
def __build_backend_wrapper_class(backend: Backend, registry: Registry):
    return registry.module_dict[backend.value]

#定义bakcend的包装器
BAKCEND_WRAPPER = Registry('backend', __build_backend_wrapper_class)

def get_backend_wrapper_class(backend : Backend) -> type:

    return BAKCEND_WRAPPER.build(backend)

def get_backend_file_count(backend: Backend):
    backend_class = get_backend_wrapper_class(backend)
    return backend_class,get_backend_file_count()