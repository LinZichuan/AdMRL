import math
import numpy as np

from rllab.envs.base import Step
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides

import tensorflow as tf
from slbo.utils.flags import FLAGS
from rllab.envs.mujoco.task_param import TaskNet, TaskConfig
from . import register_env

class AntTaskConfig(object):
    goal_velocity: float

    def __init__(self, n_params=1, lo=-3.0, hi=3.0):
        TaskConfig.__init__(self, n_params, lo, hi, lo, hi)
        self.goal_velocity = 1.0
        self.fixed_velocities = np.linspace(lo, hi, 40)

    def sample_goal_velocity(self, lo=-3.0, hi=3.0) -> float:
        return np.random.uniform(self.lo, self.hi)

    def sample(self, adv=False) -> None:
        if adv:
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        else:
            new_params = np.random.uniform([-3.0], [3.0])
            self.set_parameters(new_params)
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)

    def __str__(self):
        if self.goal_velocity == -math.inf:
            return 'Goal Velocity = BACKWARD (-inf)'
        elif self.goal_velocity == math.inf:
            return 'Goal Velocity = FORWARD (+inf)'
        else:
            return f'Goal Velocity = {self.goal_velocity:.4f}'

@register_env('ant-vel-rllab')
class AntTaskEnv(AntEnv):
    """
        An ant environment with a configurable goal velocity.

        The velocity is really a speed.
    """
    _task_config: AntTaskConfig

    def __init__(self, task={}, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs["task_config"]
            del kwargs['task_config']
        else:
            self._task_config = None

        super(AntTaskEnv, self).__init__(*args, **kwargs)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0]['goal_vel']

    def get_all_task_idx(self): # adapt for pearl
        return range(len(self.tasks))

    def reset_task(self, idx): # adapt for pearl
        self._task = self.tasks[idx]
        self._goal_vel = self._task['goal_vel']
        if self._task_config == None:
            self._task_config = AntTaskConfig()
        self._task_config.goal_velocity = self._goal_vel
        self.reset()

    def sample_tasks(self, num_tasks): # adapt for pearl
        goal_vels = np.random.uniform(-3.0, 3.0, (num_tasks, ))
        tasks = [{'goal_vel': goal_vel} for goal_vel in goal_vels]
        #print (tasks)
        return tasks

    @overrides
    def step(self, action):
        """
        Same as AntEnv except forward_reward is |actual - target| + 1, rather than actual.
        Cross-reference with Chelsea's implementation, in particular forward_reward computaion:
        https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/ant_env_rand.py#L52

        The special case of goal_velocity +/- inf corresponds to ant_env_rand_direc.py environment.
        See: https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/ant_env_rand_direc.py#L51
        """
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        if self._task_config.goal_velocity == -math.inf:
            forward_reward = -1 * comvel[0]
        elif self._task_config.goal_velocity == math.inf:
            forward_reward = comvel[0]
        else:
            forward_reward = -np.abs(comvel[0] - self._task_config.goal_velocity) + 1.0 
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)
