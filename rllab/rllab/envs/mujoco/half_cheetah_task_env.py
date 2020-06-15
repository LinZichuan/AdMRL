import math
import numpy as np

from rllab.envs.base import Step
#from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from gym.envs.mujoco import HalfCheetahEnv

from rllab.misc.overrides import overrides
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from lunzi.nn.parameter import Parameter
from lunzi import nn
from slbo.utils.flags import FLAGS
from rllab.envs.mujoco.task_param import TaskNet, TaskConfig
from . import register_env

class HalfCheetahTaskConfig(TaskConfig):
    """
    Remember, goal_velocity is really a speed, but we follow terminology of the MAML paper.
    """
    goal_velocity: float

    def __init__(self, n_params=1, lo=[-2.0], hi=[2.0]):
        TaskConfig.__init__(self, n_params, lo, hi, highdim=False, n_dim=n_params)
        self.goal_velocity = 1.0
        #self.fixed_velocities = np.linspace(lo, hi, 40)

    def sample(self, adv=False) -> None:
        if adv:
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        else:
            new_params = np.random.uniform([self.lo], [self.hi])
            self.set_parameters(new_params)
            self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)

    def __str__(self):
        if self.goal_velocity == -math.inf:
            return 'Goal Velocity = BACKWARD (-inf)'
        elif self.goal_velocity == math.inf:
            return 'Goal Velocity = FORWARD (+inf)'
        else:
            return f'Goal Velocity = {self.goal_velocity:.4f}'

@register_env('cheetah-vel-rllab')
class HalfCheetahTaskEnv(HalfCheetahEnv):
    """
        A half cheetah environment with a configurable goal velocity.
    """
    _task_config: HalfCheetahTaskConfig

    def __init__(self, task={}, n_tasks=100, randomize_tasks=False, *args, **kwargs):
        if 'task_config' in kwargs:
            self._task_config = kwargs['task_config']
            del kwargs['task_config']
        else:
            self._task_config = None

        super(HalfCheetahTaskEnv, self).__init__(*args, **kwargs)

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

    @overrides
    def step(self, action):
        """
        Same as HalfCheetahEnv except run_cost is |actual - target| rather than actual.
        Cross-reference with Chelsea's implementation, in particular run_cost computation:
        https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/half_cheetah_env_rand.py#L62

        The special case of goal_velocity +/- inf corresponds to half_cheetah_env_rand_direc.py environment.
        See: https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/half_cheetah_env_rand_direc.py#L70
        """
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))

        if self._task_config.goal_velocity == -math.inf:
            run_cost = self.get_body_comvel("torso")[0]
        elif self._task_config.goal_velocity == math.inf:
            run_cost = -1 * self.get_body_comvel("torso")[0]
        else:
            if FLAGS.task.reward == 'abs':
                run_cost = np.abs(self.get_body_comvel("torso")[0] - self._task_config.goal_velocity)
            elif FLAGS.task.reward == 'quadratic':
                run_cost = np.square(self.get_body_comvel("torso")[0] - self._task_config.goal_velocity)
            else:
                raise Exception(f'{FLAGS.task.reward} reward function is not available!')
        ctrl_cost = ctrl_cost.astype(np.float32)
        run_cost = run_cost.astype(np.float32)
        #print (ctrl_cost.dtype, run_cost.dtype)
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        reward_state = np.array(self.get_body_comvel("torso")[0]).reshape((1,))
        return Step(next_obs, reward, done, reward_ctrl=-ctrl_cost, reward_state=reward_state)
