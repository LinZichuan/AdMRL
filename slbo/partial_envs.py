import numpy as np
from slbo.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.ant_env import AntEnv
from slbo.envs.mujoco.hopper_env import HopperEnv
from slbo.envs.mujoco.walker_env import WalkerEnv
from slbo.envs.mujoco.ant_task_env import AntTaskEnv, AntTaskConfig
from slbo.envs.mujoco.ant3d_task_env import Ant3DTaskEnv, Ant3DTaskConfig
from slbo.envs.mujoco.ant2d_task_env import Ant2DTaskEnv, Ant2DTaskConfig
from slbo.envs.mujoco.half_cheetah_task_env import HalfCheetahTaskEnv, HalfCheetahTaskConfig
from slbo.envs.mujoco.half_cheetah_linearstate_env import HalfCheetahLinearStateEnv, HalfCheetahLinearStateConfig
from slbo.envs.mujoco.half_cheetah_goalstate_env import HalfCheetahGoalStateEnv, HalfCheetahGoalStateConfig
from slbo.envs.mujoco.half_cheetah_2d_env import HalfCheetah2DEnv, HalfCheetah2DConfig
from slbo.envs.mujoco.hopper_2d_env import Hopper2DEnv, Hopper2DConfig
from slbo.envs.mujoco.walker_2d_env import Walker2DEnv, Walker2DConfig
from slbo.envs.mujoco.humanoid_task_env import HumanoidTaskEnv, HumanoidTaskConfig

task_envs = {
    'AntTask-v2': AntTaskEnv,
    'Ant3DTask-v2': Ant3DTaskEnv,
    'Ant2DTask-v2': Ant2DTaskEnv,
    'HalfCheetahTask-v2': HalfCheetahTaskEnv,
    'HalfCheetahLinearState-v2': HalfCheetahLinearStateEnv,
    'HalfCheetahGoalState-v2': HalfCheetahGoalStateEnv,
    'HalfCheetah2D-v2': HalfCheetah2DEnv,
    'Hopper2D-v2': Hopper2DEnv,
    'Walker2D-v2': Walker2DEnv,
    'HumanoidTask-v2': HumanoidTaskEnv,
}
task_configs = {
    'AntTask-v2': AntTaskConfig,
    'AntHalfFrictionTask-v2': AntTaskConfig,
    'AntMissingLegTask-v2': AntTaskConfig,
    'Ant3DTask-v2': Ant3DTaskConfig,
    'Ant2DTask-v2': Ant2DTaskConfig,
    'HalfCheetahTask-v2': HalfCheetahTaskConfig,
    'HalfCheetahLinearState-v2': HalfCheetahLinearStateConfig,
    'HalfCheetahGoalState-v2': HalfCheetahGoalStateConfig,
    'HalfCheetah2D-v2': HalfCheetah2DConfig,
    'Hopper2D-v2': Hopper2DConfig,
    'Walker2D-v2': Walker2DConfig,
    'HumanoidTask-v2': HumanoidTaskConfig,
}

for k in task_envs:
    assert k in task_configs

def make_env(id: str, task_config=None):
    if id in task_envs:
        env = task_envs[id](task_config=task_config)
    else:
        raise Exception(f"env {id} not recognized")
        
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    return env

def make_task(id: str):
    if id not in task_envs:
        raise Exception(f"env {id} not recognized")

    return task_configs[id]()
