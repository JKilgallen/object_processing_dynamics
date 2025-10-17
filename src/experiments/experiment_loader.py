import yaml
from importlib import import_module

class DynamicLoader(yaml.FullLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor('!eval', self.construct_eval)
        self.add_constructor('!call', self.construct_call)

    def construct_call(self, _, node):
        func, *kwargs = self.construct_scalar(node).split(' kwargs=')
        kwargs = eval(*kwargs) if len(kwargs) > 0 else {}
        if '.' in func:
            
            *module_name, attr = func.split('.')
            module_name = '.'.join(module_name) if len(module_name) > 0 else None
            try:
                module = import_module(module_name)
            except:
                raise ValueError(f'Could not import {module_name}')
            try:
                func = getattr(module, attr)
            except:
                raise ValueError(f'{attr} is not a valid attribute of {module}')
        else:
            try:
                func = eval(func)
            except:
                raise ValueError(f'{func} is not a valid expression')
            
        return func(**kwargs)

    def construct_eval(self, _, node):
        value = self.construct_scalar(node)
        
        if '.' in value:
            
            *module_name, attr = value.split('.')
            module_name = '.'.join(module_name) if len(module_name) > 0 else None
            try:
                module = import_module(module_name)
            except:
                raise ValueError(f'Could not import {module_name}')
            try:
                return getattr(module, attr)
            except:
                raise ValueError(f'{attr} is not a valid attribute of {module}')
        else:
            try:
                value = eval(value)
            except:
                raise ValueError(f'{value} is not a valid expression')

        return value
    
class FlexibleSafeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_multi_constructor("!", self._ignore_unknown_tag)

    def _ignore_unknown_tag(self, loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        elif isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return loader.construct_object(node)
    
def load_experiment_cfg(experiment_id, safe=False):
    with open(f'config/experiments/{experiment_id}.yaml', 'r') as file:
        experiment = yaml.load(file, Loader=FlexibleSafeLoader if safe else DynamicLoader)

    return dict(experiment)

def load_dataset_cfg(dataset_id, safe=False):
    with open(f'config/datasets/{dataset_id}.yaml', 'r') as file:
        dataset_cfg = yaml.load(file, Loader=FlexibleSafeLoader if safe else DynamicLoader)
    return dict(dataset_cfg)