import math
import numpy as np

from rllab.envs.base import Step
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides

import tensorflow as tf
from slbo.utils.flags import FLAGS
from rllab.envs.mujoco.task_param import TaskNet, TaskConfig
from . import register_env

class Ant3DTaskConfig(TaskConfig):
    goal_velocity: np.array

    def __init__(self, n_params=3, lo=np.array([-3.0,-3.0,0.4]), hi=np.array([3.0,3.0,0.6]), coef=np.array([1.,1.,30.]), func='abs', n_dim=3, _init=True):
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

@register_env('ant3d-gym')
class Ant3DTaskEnv(AntEnv):
    """
        An ant environment with a configurable goal velocity.
    """
    _task_config: Ant3DTaskConfig

    def __init__(self, task={}, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs["task_config"]
            del kwargs['task_config']
        else:
            self._task_config = Ant3DTaskConfig(_init=False)
            self._task_config.goal_velocity = np.array([0, 0, 0.5])

        super(Ant3DTaskEnv, self).__init__(*args, **kwargs)

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0]['goal_vel']

    def get_all_task_idx(self): # adapt for pearl
        return range(len(self.tasks))

    def reset_task(self, idx): # adapt for pearl
        self._task = self.tasks[idx]
        self._goal_vel = self._task['goal_vel']
        if self._task_config == None:
            self._task_config = Ant3DTaskConfig(_init=False)
        self._task_config.goal_velocity = self._goal_vel
        self.reset()

    def set_task(self, task): # adapt for pearl
        self._task_config.goal_velocity = task
        self.reset()

    def get_all_eval_tasks(self):
        self.eval_tasks = []
        for a in np.linspace(-3.0, 3.0, 4):
            for b in np.linspace(-3.0, 3.0, 4):
                for c in np.linspace(0.4, 0.6, 3):
                    self.eval_tasks.append([a,b,c])
        print ('#eval_tasks:', len(self.eval_tasks))
        return self.eval_tasks
        #self.eval_tasks = []
        #for a in np.linspace(-3.0, 3.0, 10):
        #    for b in np.linspace(-3.0, 3.0, 10):
        #        #for c in np.linspace(0.4, 0.6, 3):
        #        self.eval_tasks.append([a,b,0.5])
        #print ('#eval_tasks:', len(self.eval_tasks))
        #return self.eval_tasks

    def sample_tasks(self, num_tasks): # adapt for pearl
        x = np.random.uniform(-3.0, 3.0, (num_tasks, 1))
        y = np.random.uniform(-3.0, 3.0, (num_tasks, 1))
        z = np.random.uniform(0.4, 0.6, (num_tasks, 1))
        goal_vels = np.concatenate([x,y,z], axis=1)  #(num_tasks, 3)
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
        #zpos = self.get_body_com("torso")[2]
        #print (f"comvel={comvel}, com={com}")

        next_obs = self.get_current_obs()
        zpos = next_obs[2]
        #print (next_obs.tolist())
        #print (zpos, next_obs[-4], com[2])
        #print ('diff=', zpos-next_obs[-4])
        #assert (zpos == next_obs[-4])

        action = np.clip(action, *self.action_bounds)
        reward_ctrl = -1e-1 * 0.5 * np.sum(np.square(action))

        forward_reward = -np.abs(comvel[0] - self._task_config.goal_velocity[0]) * self._task_config.coef[0]
        forward_reward += -np.abs(comvel[1] - self._task_config.goal_velocity[1]) * self._task_config.coef[1]
        forward_reward += -np.abs(zpos - self._task_config.goal_velocity[2]) * self._task_config.coef[2]
        entry = 3.0

        reward_state = np.array([comvel[0], comvel[1], zpos])
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
