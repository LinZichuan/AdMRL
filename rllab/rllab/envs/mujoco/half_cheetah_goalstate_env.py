import math
import os
import numpy as np

from rllab.envs.base import Step
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv

from rllab.misc.overrides import overrides
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from lunzi.nn.parameter import Parameter
from lunzi import nn
from slbo.utils.flags import FLAGS
from rllab.envs.mujoco.task_param import TaskNet, TaskConfig
from . import register_env
import pickle

class HalfCheetahGoalStateConfig(TaskConfig):
    """
    Remember, goal_velocity is really a speed, but we follow terminology of the MAML paper.
    """
    goal_velocity: float

    def __init__(self, n_params=18, load_state=True):
        self.func = 'abs'
        if load_state == True:
            data = pickle.load(open(f'{os.getcwd()}/HCstate-state.pkl', 'rb'))
            s_min, s_max = data['min'], data['max']
            self.s_mean, self.s_std = data['mean'], data['std']
            for i in range(s_min.shape[0]):
                #if s_max[i] - s_min[i] == 0:
                #    assert(False)
                s_max[i] = s_max[i] * 5
                s_min[i] = s_min[i] * 5
            lo, hi = np.array([-3.0]*n_params), np.array([3.0]*n_params)
        else:
            lo, hi = np.array([-5.0]*n_params), np.array([5.0]*n_params)
        self.coef = np.array([1.0] * n_params)
        print (f'coef = {self.coef}')

        TaskConfig.__init__(self, n_params, lo, hi)

    def sample(self, adv=False) -> None:
        if adv:
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        else:
            new_params = np.random.uniform(self.lo, self.hi)
            self.set_parameters(new_params)
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity:.4f}'

#@register_env('cheetah-state-rllab')
class HalfCheetahGoalStateEnv(HalfCheetahEnv):
    """
        A half cheetah environment with a configurable goal velocity.
    """
    _task_config: HalfCheetahGoalStateConfig

    def __init__(self, task={}, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs['task_config']
            del kwargs['task_config']
        else:
            self._task_config = None

        super(HalfCheetahGoalStateEnv, self).__init__(*args, **kwargs)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0]['goal_vel']

        self.stat = []

    def get_all_task_idx(self): # adapt for pearl
        return range(len(self.tasks))

    def reset_task(self, idx): # adapt for pearl
        self._task = self.tasks[idx]
        self._goal_vel = self._task['goal_vel']
        if self._task_config == None:
            self._task_config = HalfCheetahTaskConfig()
        self._task_config.goal_velocity = self._goal_vel
        self.reset()

    def sample_tasks(self, num_tasks): # adapt for pearl
        tasks = [{'goal_vel': np.random.uniform(self._task_config.lo, self._task_config.hi)} for _ in range(num_tasks)]
        return tasks

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)

        reward_ctrl = -1e-1 * 0.5 * np.sum(np.square(action))

        norm_next_obs = (next_obs - self._task_config.s_mean) / self._task_config.s_std
        reward_run = np.sum(-np.abs(norm_next_obs - self._task_config.goal_velocity) * self._task_config.coef)
        reward_run = reward_run.astype(np.float32)

        reward = reward_ctrl + reward_run
        done = False
        reward_state = norm_next_obs
        return Step(next_obs, reward, done, reward_ctrl=reward_ctrl, reward_state=reward_state)
