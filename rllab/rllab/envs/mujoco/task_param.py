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

class TaskNet(nn.Module):

    def __init__(self, n_params):
        '''
        self.goal_velocity:  [-1,1]
        '''
        super().__init__()
        with self.scope:
            layers = []
            layers.append(nn.Linear(1, n_params, bias=False, weight_initializer=normc_initializer(1.0)))
            if FLAGS.task.scaler == 'tanh':
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        c = tf.constant(1, shape=[1,1], dtype=tf.float32)
        #self.net(c) -> [None, n_params]
        if n_params == 1:
            self.goal_velocity = self.net(c)[0]
        else:
            self.goal_velocity = self.net(c)[0]
        print (self.goal_velocity)

class TaskConfig(object):

    def __init__(self, n_params, lo, hi, n_dim=2, scale_psi=True, _init=True):
        self.n_params = n_params
        self.n_dim = n_params
        self.lo, self.hi = lo, hi
        self.net = TaskNet(n_params)
        print ("tasknet:", self.net.parameters()[0])
        x = self.net.goal_velocity  # [-1,1]
        if FLAGS.task.scaler == 'clip':
            x = x - tf.stop_gradient(x) + tf.clip_by_value(x, -1.0, 1.0)
            if scale_psi:
                print ('scale_psi to [lo,hi]')
                self.goal_velocity_params = (x + 1.0) / 2.0 * (self.hi - self.lo) + self.lo
            else:
                self.goal_velocity_params = x
        print (f'goalname: {FLAGS.task.goalname}')
        if n_params == 1:
            self.goal_velocity_params = tf.reshape(self.goal_velocity_params, (-1,))
        if _init:
            self.init()

    def reduce_dim(self):
        pass

    def init(self, params=None):
        if params is not None:
            self.set_parameters(params)
        else:
            print ("lo&hi=", self.lo, self.hi)
            params = np.array([0.3] * self.n_params)
            self.set_parameters(params)

    def random_sample(self, samplemethod='uniform'):
        print (f"current task goal velocity = {tf.get_default_session().run(self.goal_velocity_params)}")
        if FLAGS.task.scaler == 'clip':
            if samplemethod == 'uniform':
                new_params = np.random.uniform(-1.0, 1.0, self.n_params) #this is correct because we do not tanh
                self.set_parameters(new_params)
                print (f"uniformly sample from U(-1.0, 1.0), new_params={new_params}, get goal_vel = {self.goal_velocity}")
            elif samplemethod == 'normal':
                new_params = np.random.normal(0, 1.0, self.n_params)
                self.set_parameters(new_params)
                print (f"sample from gaussian N(0, 1.0), new_params={new_params}, get goal_vel = {self.goal_velocity}")
            else:
                assert (False)

    def parameters(self):
        params = self.net.parameters()[0]
        return params

    def set_parameters(self, new_params):
        new_params = np.array(new_params)
        print (new_params)
        new_params = new_params.reshape(-1)
        if FLAGS.task.scaler == 'clip':
            new_params = np.clip(new_params, -1.0, 1.0)
        else:
            new_params = np.clip(new_params, -4.0, 4.0)
        new_params = np.reshape(new_params, (1, self.n_dim))
        tf.get_default_session().run(tf.assign(self.parameters(), new_params))
        self.goal_velocity = tf.get_default_session().run(self.goal_velocity_params)
        self.params_ = new_params
        print (f'set_parameters(): params={new_params}, goal-vel={self.goal_velocity}')
