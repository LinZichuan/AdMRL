#from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math

from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, **kwargs):
        self.comvel = np.array([0., 0., 0.])
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 1)
        utils.EzPickle.__init__(self)

    def get_current_obs(self):
        return self._get_obs()

    def _get_obs(self):
        #print ("com:", self.get_body_com("torso").shape)
        #print ("comvel:", self.get_body_comvel("torso").shape)
        return np.concatenate([
            self.sim.data.qpos.flat,  # 15
            self.sim.data.qvel.flat,  # 14
            #self.get_body_xmat("torso").flat,  # 9
            self.get_body_com("torso"),  # 3
            self.get_body_comvel("torso"),  # 3
        ]).reshape(-1)
        #return np.concatenate([
        #    self.sim.data.qpos.flat,
        #    self.sim.data.qvel.flat,
        #    np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        #    self.get_body_xmat("torso").flat,
        #    self.get_body_com("torso"),
        #]).reshape(-1)

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
        return self.sim.data.body_xmat[idx].reshape((3, 3))

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
        ob = self.get_current_obs()
        reward = 0.
        done = False
        return Step(ob, float(reward), done)

        #comvel = self.get_body_comvel("torso")
        #forward_reward = comvel[0]
        #lb, ub = self.action_bounds
        #scaling = (ub - lb) * 0.5
        #ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        #contact_cost = 0.5 * 1e-3 * np.sum(
        #    np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        #survive_reward = 0.05
        #reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        #state = self._state
        #notdone = np.isfinite(state).all() \
        #    and state[2] >= 0.2 and state[2] <= 1.0
        #done = not notdone
        #ob = self.get_current_obs()
        #return Step(ob, float(reward), done)

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

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

