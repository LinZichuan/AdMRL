import math
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

class HalfCheetah2DConfig(TaskConfig):
    """
    Remember, goal_velocity is really a speed, but we follow terminology of the MAML paper.
    """
    goal_velocity: float

    def __init__(self, n_params=2, lo=np.array([-2.0, 0.2]), hi=np.array([2.0, 1.0])):
        TaskConfig.__init__(self, n_params, lo, hi)
        self.coef = np.array([1.0, 1.0])
        self.func = 'abs'
        print (f'reward coef = {self.coef}')
        print (f'reward func = {self.func}')

    def sample(self, adv=False) -> None:
        if adv:
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        else:
            new_params = np.random.uniform([self.lo], [self.hi])
            self.set_parameters(new_params)
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity:.4f}'

@register_env('cheetah2d')
class HalfCheetah2DEnv(HalfCheetahEnv):
    """
        A half cheetah environment with a configurable goal velocity.
    """
    _task_config: HalfCheetah2DConfig

    def __init__(self, task={}, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs['task_config']
            del kwargs['task_config']
        else:
            self._task_config = None

        super(HalfCheetah2DEnv, self).__init__(*args, **kwargs)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0]['goal_vel']

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
        goal_vels = np.random.uniform(-2.0, 2.0, (num_tasks, ))
        tasks = [{'goal_vel': goal_vel} for goal_vel in goal_vels]
        #print (tasks)
        return tasks

    def _get_obs(self):
        self.qpos_len = len(self.sim.data.qpos.flat[1:])
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            #self.get_body_com("torso").flat,
            #self.get_body_comvel("torso").flat,
        ])

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        # reward_ctrl
        reward_ctrl = -0.1 * np.sum(np.square(action))

        # reward_run
        height, vel = next_obs[0], next_obs[self.qpos_len]
        #print (f'h={height}, vel={vel}, len={self.qpos_len}')
        reward_run = -np.abs(vel - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        reward_run += -np.abs(height - self._task_config.goal_velocity[1]) * self._task_config.coef[1]

        # reward, done, reward_state
        reward = reward_ctrl + reward_run
        done = False
        reward_state = np.array([vel, height])
        return Step(next_obs, reward, done, reward_ctrl=reward_ctrl, reward_state=reward_state)
