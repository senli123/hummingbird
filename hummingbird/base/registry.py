"""定义注册器
"""
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

        
