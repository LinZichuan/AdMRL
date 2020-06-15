import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
#from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides

from gym import utils
from gym.envs.mujoco import mujoco_env


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)

    def get_current_obs(self):
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            #self.get_body_com("torso").flat,
            #self.get_body_comvel("torso").flat,
        ])

    def reset_model(self):
        self.comvel = np.array([0., 0., 0.])
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    @property
    def action_bounds(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        return bounds.T

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.sim.data.xmat[idx].reshape((3, 3))

    def get_body_comvel(self, body_name):
        return self.comvel

    def forward_dynamics(self, action):
        # https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_direc_2d.py#L35
        xposbefore = np.copy(self.get_body_com('torso'))
        self.do_simulation(action, 1)
        xposafter = self.get_body_com('torso')
        #print (xposbefore, xposafter, self.dt)
        self.comvel = (xposafter - xposbefore) / self.dt
        #print ("comvel:", self.comvel)

    def step(self, action):
        self.forward_dynamics(action)
        #self.do_simulation(action, 1)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = 0. #-1 * self.get_body_comvel("torso")[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return Step(next_obs, reward, done)

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
