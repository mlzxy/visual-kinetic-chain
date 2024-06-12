#!/usr/bin/python3
from datetime import datetime
import re
import os.path as osp
import os 
import jstyleson as json
import sys
import importlib.util
import copy
from easydict import EasyDict as dict_to_class

_global_ = {
    'active_exp_id': '',
    'naming_function': None,
    'config': None
}


def require(path, toplevel=False, name=None, auto_name_with_parent=False):
    """
    support loads json and python
    with suffix as run id
    """
    path = str(path)
    # transform path to absolute path
    if not path.startswith('/'):
        parent_scope = sys._getframe(1).f_globals
        if '__file__' in parent_scope:
            parent_path = parent_scope['__file__']
            caller_folder = osp.dirname(parent_path)
        else:
            caller_folder = osp.abspath('')
        path = osp.abspath(osp.join(caller_folder, path))

    # get run_id 
    folder = osp.dirname(path)
    bname = osp.basename(path)

    current_time = datetime.now()
    run_id = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    if '#' in bname:
        bname, run_id = bname.split('#')
        path = osp.join(folder, bname)
    
    # guess ext
    if osp.isdir(path):
        path = osp.join(path, "base")
    _, ext = osp.splitext(path)
    if len(ext) == 0:
        if osp.exists(path + ".py"):
            path += '.py'
        elif osp.exists(path + '.json'):
            path += '.json'
        else:
            raise FileNotFoundError(path + ".py/.json")
    
    assert osp.exists(path), f'config {path} is not found!'
    is_py = path.endswith('.py')
    if is_py:
        module_name = re.sub('/+', '/', path.strip('.py')).replace('/', '.')
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        with open(path, 'r') as f:
            mod = dict_to_class(json.load(f))

    if toplevel:
        conf = copy.deepcopy(mod.Config if is_py else mod)
        if hasattr(mod, 'Name'): 
            _global_['naming_function'] = mod.Name
        if name is not None:
            setattr(conf, 'name', name)
        else:
            if not hasattr(conf, 'name'):
                auto_name =  bname.split('.py')[0]
                if auto_name_with_parent:
                    auto_name = osp.basename(folder) + '/' +  auto_name
                setattr(conf, 'name', auto_name.split('.py')[0])
        conf.name += ('/' + str(run_id))
        _global_['active_exp_id'] = conf.name
        _global_['config'] = conf
        return conf
    else:
        return mod
        
        
def get_exp_id():
    if _global_['naming_function'] is not None:
        return _global_['naming_function'](_global_['config'])
    else:
        return _global_['active_exp_id']

def write_cfg(path: str, data: object):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)