import math
import numpy as np

from rllab.envs.base import Step
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides

import tensorflow as tf
from slbo.utils.flags import FLAGS
from rllab.envs.mujoco.task_param import TaskNet, TaskConfig
from . import register_env

class Ant2DTaskConfig(TaskConfig):
    goal_velocity: np.array

    def __init__(self, n_params=2, lo=np.array([-3.0,-3.0]), hi=np.array([3.0,3.0]), coef=np.array([1.,1.]), func='abs', n_dim=2, _init=True):
        TaskConfig.__init__(self, n_params, lo, hi, _init=_init)
        self.coef = coef
        self.func = func
        print (f'reward coef = {coef}')
        print (f'reward func = {func}')

    def sample(self, adv=False) -> None:
        if adv:
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        else:
            new_params = np.random.uniform(self.lo, self.hi)
            self.set_parameters(new_params)
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)

    def __str__(self):
        return f'Goal Velocity = {self.goal_velocity}'

@register_env('ant2d-gym')
class Ant2DTaskEnv(AntEnv):
    """
        An ant environment with a configurable goal velocity.
    """
    _task_config: Ant2DTaskConfig

    def __init__(self, task={}, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs["task_config"]
            del kwargs['task_config']
        else:
            #self._task_config = None
            self._task_config = Ant2DTaskConfig(_init=False)
            self._task_config.goal_velocity = np.array([0, 0])

        super(Ant2DTaskEnv, self).__init__(*args, **kwargs)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0]['goal_vel']

    def get_all_task_idx(self): # adapt for pearl
        return range(len(self.tasks))

    def reset_task(self, idx): # adapt for pearl
        self._task = self.tasks[idx]
        self._goal_vel = self._task['goal_vel']
        if self._task_config == None:
            self._task_config = Ant2DTaskConfig(_init=False)
        self._task_config.goal_velocity = self._goal_vel
        self.reset()

    def sample_tasks(self, num_tasks): # adapt for pearl
        goal_vels = np.random.uniform(-3.0, 3.0, (num_tasks, 2))  #(num_tasks, 2)
        tasks = [{'goal_vel': goal_vel} for goal_vel in goal_vels]
        #print (tasks)
        return tasks

    @overrides
    def step(self, action):
        """
        Same as AntTaskEnv except two-dimensional goal velocity.
        """
        self.forward_dynamics(action)
        #self.do_simulation(action, self.frame_skip)
        comvel = self.get_body_comvel("torso")
        com = self.get_body_com("torso")

        next_obs = self.get_current_obs()

        action = np.clip(action, *self.action_bounds)
        reward_ctrl = -1e-1 * 0.5 * np.sum(np.square(action))

        forward_reward = -np.abs(comvel[0] - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        forward_reward += -np.abs(comvel[1] - self._task_config.goal_velocity[1]) * self._task_config.coef[1]
        entry = 5.0

        reward_state = np.array([comvel[0], comvel[1]])
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.
        survive_reward = 0.05
        reward = forward_reward - (ctrl_cost + contact_cost - survive_reward - entry) 
        ctrl_cost = ctrl_cost + contact_cost - survive_reward - entry
        notdone = np.isfinite(next_obs).all() \
            and next_obs[2] >= 0.2 and next_obs[2] <= 1.0
        done = not notdone
        return Step(next_obs, float(reward), done, reward_ctrl=-ctrl_cost, reward_state=reward_state)
