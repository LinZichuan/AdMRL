import os
import importlib


rllab_ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in rllab_ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        rllab_ENVS[name] = fn
        return fn

    return register_env_fn


## automatically import any envs in the envs/ directory
#print (__file__)
#for file in os.listdir(os.path.dirname(__file__)):
#    if file.endswith('.py') and not file.startswith('_'):
#        module = file[:file.find('.py')]
#        if 'rand_params' in module: continue
#        print ('rllab.envs.mujoco.' + module)
#        importlib.import_module('rllab.envs.mujoco.' + module)
