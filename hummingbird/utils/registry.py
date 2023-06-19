"""定义注册器
"""
import inspect
import warnings
from functools import partial

def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')
class Registry:

    def __init__(self, name, build_func = None):
        """注册器的初始化函数

        Args:
            name (_type_): 该注册器的名字
            build_func (_type_, optional): 该注册器构建函数的方法
        """
        self.name = name
        self._module_dict = dict()
        if build_func is None:
            self.build_func = build_from_cfg
        else:
            self.build_func = build_func

        def __len__(self):
            return len(self._module_dict)

        def __contains__(self, key):
            return self.get(key) is not None
        
        def __repr__(self):
            format_str = self.__class__.__name__ + \
                            f'(name = {self._name},' \
                            f'items= {self._module_dict})'
            return format_str

        @property
        def name(self):
            return self._name
        
        @property
        def module_dict(self):
            return self._module_dict

        
        def get(self, key):
            if key in self._module_dict:
                return self._module_dict[key]
            else:
                return None

        def build(self, *args, **kwargs):
            return self.build_func(*args, **kwargs, registry = self)
        

        def _register_module(self, module_class, module_name=None, force=False):
            if not inspect.isclass(module_class):
                raise TypeError('module must be a class, '
                                f'but got {type(module_class)}')

            if module_name is None:
                module_name = module_class.__name__
            if isinstance(module_name, str):
                module_name = [module_name]
            for name in module_name:
                if not force and name in self._module_dict:
                    raise KeyError(f'{name} is already registered '
                                f'in {self.name}')
                self._module_dict[name] = module_class

        def register_module(self, name=None, force=False, module=None):
            if not isinstance(force, bool):
                raise TypeError(f'force must be a boolean, but got {type(force)}')
            # NOTE: This is a walkaround to be compatible with the old api,
            # while it may introduce unexpected bugs.
            if not isinstance(name, str):
                raise TypeError(
                    'name must be either of None, an instance of str or a sequence'
                    f'  of str, but got {type(name)}')

            # use it as a normal method: x.register_module(module=SomeClass)
            if module is not None:
                self._register_module(
                    module_class=module, module_name=name, force=force)
                return module

            # use it as a decorator: @x.register_module()
            def _register(cls):
                self._register_module(
                    module_class=cls, module_name=name, force=force)
                return cls

            return _register

        
