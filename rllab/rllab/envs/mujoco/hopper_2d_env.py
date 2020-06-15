import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
#from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.hopper_env import HopperEnv
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from lunzi.nn.parameter import Parameter
from lunzi import nn
from slbo.utils.flags import FLAGS
from rllab.envs.mujoco.task_param import TaskNet, TaskConfig
from . import register_env

# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)

class Hopper2DConfig(TaskConfig):
    goal_velocity: float

    def __init__(self, n_params=2, lo=np.array([-2.0, 1.2]), hi=np.array([2.0, 2.0]), _init=True):
        TaskConfig.__init__(self, n_params, lo, hi, _init=_init)
        self.coef = np.array([1.0, 5.0])
        self.func = 'abs'
        print (f'reward coef = {self.coef}')
        print (f'reward func = {self.func}')
        self.alive_bonus = 10.

    def sample(self, adv=False) -> None:
        if adv:
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        else:
            new_params = np.random.uniform([self.lo], [self.hi])
            self.set_parameters(new_params)
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity:.4f}'

@register_env('hopper2d-gym')
class Hopper2DEnv(HopperEnv):
    task_config: Hopper2DConfig

    def __init__(self, task={}, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs['task_config']
            del kwargs['task_config']
        else:
            #self._task_config = None
            self._task_config = Hopper2DConfig(_init=False)
            self._task_config.goal_velocity = np.array([0, 1.5])

        super(Hopper2DEnv, self).__init__(*args, **kwargs)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0]['goal_vel']

    def get_all_task_idx(self): # adapt for pearl
        return range(len(self.tasks))

    def reset_task(self, idx): # adapt for pearl
        self._task = self.tasks[idx]
        self._goal_vel = self._task['goal_vel']
        if self._task_config == None:
            self._task_config = Hopper2DConfig(_init=False)
        self._task_config.goal_velocity = self._goal_vel
        self.reset()

    def sample_tasks(self, num_tasks): # adapt for pearl
        x = np.random.uniform(-2.0, 2.0, (num_tasks, 1))
        z = np.random.uniform(1.2, 2.0, (num_tasks, 1))
        goal_vels = np.concatenate([x,z], axis=1)  #(num_tasks, 2)
        tasks = [{'goal_vel': goal_vel} for goal_vel in goal_vels]
        #print (tasks)
        return tasks

    @overrides
    def _get_obs(self):
        #print ('get from here')
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            #self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10),
        ])

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        # posafter, height, ang = self.sim.data.qpos[0:3]
        # height, ang = next_obs[0:2]

        # reward_ctrl
        reward_ctrl = - 0.001 * np.sum(np.square(action)) + self._task_config.alive_bonus
        # reward_run
        height, ang, vel = next_obs[0], next_obs[1], next_obs[5]
        #print ("qpos:", self.sim.data.qpos.shape)
        #print (next_obs.shape, vel, height)
        reward_run = -np.abs(vel - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        reward_run += -np.abs(height - self._task_config.goal_velocity[1]) * self._task_config.coef[1]
        reward_state = np.array([vel, height])

        # reward, done, reward_state
        reward = reward_ctrl + reward_run
        done = not (np.isfinite(next_obs).all() and (np.abs(next_obs[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        #notdone = np.isfinite(next_obs).all() and \
        #    (np.abs(next_obs[3:]) < 100).all() and (next_obs[0] > .7) and \
        #    (abs(next_obs[2]) < .2)
        #done = not notdone

        return Step(next_obs, reward, done, reward_ctrl=reward_ctrl, reward_state=reward_state)
