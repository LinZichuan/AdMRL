import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
#from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides

from gym import utils
from gym.envs.mujoco import mujoco_env

def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class WalkerEnv(mujoco_env.MujocoEnv, Serializable):

    FILE = 'walker2d.xml'

    def __init__(self, ctrl_cost_coeff=1e-2, *args, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, 'walker2d.xml', 1)
        utils.EzPickle.__init__(self)
        #super(WalkerEnv, self).__init__(*args, **kwargs)
        #Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return self._get_obs()
        #return np.concatenate([
        #    self.model.data.qpos.flat,
        #    self.model.data.qvel.flat,
        #    self.get_body_com("torso").flat,
        #])

    @overrides
    def _get_obs(self):
        return None

    def reset_model(self):
        self.comvel = np.array([0., 0., 0.])
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def forward_dynamics(self, action):
        xposbefore = np.copy(self.get_body_com('torso'))
        self.do_simulation(action, 1)
        xposafter = self.get_body_com('torso')
        #print (xposbefore, xposafter, self.dt)
        self.comvel = (xposafter - xposbefore) / self.dt
        #print ("comvel:", self.comvel)

    def step(self, action):
        return None
        '''
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * \
            np.sum(np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        qpos = self.model.data.qpos
        done = not (qpos[0] > 0.8 and qpos[0] < 2.0
                    and qpos[2] > -1.0 and qpos[2] < 1.0)
        return Step(next_obs, reward, done)
        '''

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

