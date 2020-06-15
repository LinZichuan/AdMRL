import math
import numpy as np
from .simple_humanoid_env import SimpleHumanoidEnv
from gym.envs.mujoco import mujoco_env
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

import tensorflow as tf
from slbo.utils.flags import FLAGS
from rllab.envs.mujoco.task_param import TaskNet, TaskConfig
from . import register_env

class HumanoidTaskConfig(TaskConfig):
    goal_velocity: float

    def __init__(self, n_params=3, lo=np.array([-1.5,-1.5,1.2]), hi=np.array([1.5,1.5,1.6])):
        self.func = 'abs'
        self.coef = [1.0, 1.0, 7.5]
        TaskConfig.__init__(self, n_params, lo, hi)
        print (f'reward coef = {self.coef}')
        print (f'reward func = {self.func}')

    #def __init__(self, n_params=2, lo=np.array([-1.5,-1.5]), hi=np.array([1.5,1.5])):
    #    self.func = 'abs'
    #    self.coef = [1.0, 1.0]
    #    TaskConfig.__init__(self, n_params, lo, hi)
    #    print (f'reward coef = {self.coef}')
    #    print (f'reward func = {self.func}')

    def sample(self, adv=False) -> None:
        if adv:
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        else:
            new_params = np.random.uniform(self.lo, self.hi)
            self.set_parameters(new_params)
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity:.4f}'

@register_env('humanoid-vel-rllab')
class HumanoidTaskEnv(SimpleHumanoidEnv):
    _task_config: HumanoidTaskConfig

    def __init__(self, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs["task_config"]
            del kwargs['task_config']
        else:
            self._task_config = None

        super(HumanoidTaskEnv, self).__init__(*args, **kwargs)

        self._task = None
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0]['goal_vel']

    def get_all_task_idx(self): # adapt for pearl
        return range(len(self.tasks))

    def reset_task(self, idx): # adapt for pearl
        self._task = self.tasks[idx]
        self._goal_vel = self._task['goal_vel']
        if self._task_config == None:
            self._task_config = HumanoidTaskConfig()
        self._task_config.goal_velocity = self._goal_vel
        self.reset()

    def sample_tasks(self, num_tasks): # adapt for pearl
        goal_vels = np.random.uniform(-1.5, 1.5, (num_tasks, ))
        tasks = [{'goal_vel': goal_vel} for goal_vel in goal_vels]
        #print (tasks)
        return tasks
        
    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        zpos = next_obs[2]

        alive_bonus = 0.2
        data = self.sim.data

        comvel = self.get_body_comvel("torso")
        lin_vel_reward = -np.abs(comvel[0] - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        lin_vel_reward += -np.abs(comvel[1] - self._task_config.goal_velocity[1]) * self._task_config.coef[1]
        lin_vel_reward += -np.abs(zpos - self._task_config.goal_velocity[2]) * self._task_config.coef[2]
        entry = 0.8 * 3
        #entry = 0.8 * 2
        reward_state = np.array([comvel[0], comvel[1], zpos])

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 5.e-4 * np.sum(np.square(action / scaling))
        impact_cost = 0.
        vel_deviation_cost = 5.e-3 * np.sum(np.square(comvel[1:]))
        done = data.qpos.flat[2] < 0.8 or data.qpos.flat[2] > 2.0
        
        reward_ctrl = alive_bonus - ctrl_cost - impact_cost - vel_deviation_cost + entry
        reward = lin_vel_reward + reward_ctrl

        return Step(next_obs, float(reward), done, reward_ctrl=reward_ctrl, reward_state=reward_state)
