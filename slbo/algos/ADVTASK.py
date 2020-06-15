import lunzi.nn as nn
from lunzi.Logger import logger
import tensorflow as tf
import numpy as np
from typing import List, Callable
from lunzi import Tensor

from slbo.utils.runner import Runner
from slbo.envs.batched_env import BatchedEnv
from tensorflow.python.ops.parallel_for.gradients import jacobian
from slbo.algos.TRPO import TRPO
from slbo.utils.flags import FLAGS
import csv, json, pickle
from slbo.policies import BaseNNPolicy
from slbo.v_function import BaseVFunction
norm = np.linalg.norm


def average_l2_norm(x):
    return np.sqrt((x**2).mean())


# for damping, modify func_Ax
def conj_grad(mat_mul_vec: Callable[[np.ndarray], np.ndarray], b, n_iters=10, residual_tol=1e-6, verbose=False):
    print ("b.norm in CG:", norm(b))
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b) #+ 1e-4
    r_dot_r = r.dot(r)

    x_list = [x]
    x_norm_list = [np.linalg.norm(x)]
    res_list = [r_dot_r]
    res_plot = []
    res_plot.append(r_dot_r.copy())
    for i in range(n_iters):
        xAx = p.dot(mat_mul_vec(p))  #x^T A^TA x > 0
        if verbose:
            logger.info('[CG] iters = %d, |Res| = %.6f, |x| = %.6f, x^TAx = %.6f', i, r_dot_r, np.linalg.norm(x), xAx)
        z = mat_mul_vec(p)
        old_p = p
        old_r_dot_r = r_dot_r
        v = r_dot_r / p.dot(z)
        x += v * p  #generate new guess
        r -= v * z
        new_r_dot_r = r.dot(r)
        if new_r_dot_r < residual_tol:
            break
        mu = new_r_dot_r / r_dot_r
        p = r + mu * p  #generate new conjugate direction
        r_dot_r = new_r_dot_r

        x_list.append(x.copy())
        x_norm_list.append(np.linalg.norm(x))
        res_list.append(r_dot_r)
        res_plot.append(r_dot_r.copy())
    if verbose:
        logger.info('[CG] iters = %d, |Res| = %.6f, |x| = %.6f', n_iters, r_dot_r, np.linalg.norm(x))
    idx = np.argmin(res_list)
    x = x_list[idx]
    res = res_list[idx]
    print (f"res = {res}, norm = {x_norm_list[idx]}")
    print ("x = ", x)
    print ('----------------------------------------------------')
    return x, res, res_plot

class ADVTASK(nn.Module):
    def __init__(self, dim_state, dim_action, policy, vfn, warmup_policy, warmup_vfn, task, cg_damping=0.1, n_cg_iters=200, alpha=1.0, beta=1.0, nsample=8000, atype='adv'):
        super().__init__()
        self.keys = [
            'restartnext', 'goal_velocity', 'matmul','flat_grad_task' , 'flat_grad_task_opt' , 'final_grad' ,'task_params_before' , 'task_params_after' , 'returns_pre_warmup' , 
            'returns_post_warmup' , 'returns_post_slbo' , 'diff' , 'virtual_returns_post_warmup','virtual_returns_post_slbo','virtual_diff', 
            'cg_residual','val_losses_warmup' ,'val_losses_slbo', 'train_losses_warmup', 'train_losses_slbo', 'surprisal', 'warmup_policy_norm', 'policy_norm', 'logpac_norm' ,
            'pg_norm','Ax_norm', 'b_norm','x_norm','flat_grad_policy_norm','policy_loss_value','quad_loss_value','return_loss_value','return_mean_value',
            'adv_mean_value','trpo_loss','trpo_kl','trpo_g','trpo_x', 'trpo_Ax', 'trpo_res', 'alpha', 'sanitycheck'
        ]
        self.info = dict()
        for k in self.keys: self.info[k] = []
        self.info['cg_plot'] = {}
        self.info['cg_plot']['mine'] = []

        self.cg_damping = cg_damping
        self.AAx = True
        self.meanAAx = True
        self.task_num = 0
        self.advnormalize = False
        self.retnormalize = False
        self.meanret = True
        self.atype = atype # gae or ret or 1step or adv
        self.alpha = alpha
        self.beta = beta
        self.nsample = nsample
        self.task = task
        self.n_cg_iters = n_cg_iters

        self.policy = policy
        self.vfn = vfn
        self.old_policy: nn.Module = policy.clone()
        self.warmup_policy = warmup_policy
        self.old_warmup_policy: nn.Module = warmup_policy.clone()
        self.warmup_vfn = warmup_vfn
        self.warmup_policy1: nn.Module = warmup_policy.clone()
        self.warmup_policy2: nn.Module = warmup_policy.clone()
        self.warmup_policy3: nn.Module = warmup_policy.clone()

        with self.scope:
            # placeholder
            self.op_advantages = tf.placeholder(dtype=tf.float32, shape=[None], name='advantages')
            self.op_advantages_mean = tf.placeholder(dtype=tf.float32, name='advantages_mean')
            self.op_advantages_std = tf.placeholder(dtype=tf.float32, name='advantages_std')
            self.op_states = tf.placeholder(dtype=tf.float32, shape=[None, dim_state], name='states')
            self.op_actions = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name='actions')
            self.op_tangents = tf.placeholder(dtype=tf.float32, shape=[nn.utils.n_parameters(self.warmup_policy.parameters())])
            self.op_feed_params = tf.placeholder(dtype=tf.float32, shape=[None], name='feed_params')

            self.op_reward_ctrl = tf.placeholder(dtype=tf.float32, shape=[None, None], name='reward_ctrl') #nstep,nenv
            self.op_reward_state = tf.placeholder(dtype=tf.float32, shape=[None, None, self.task.n_params], name='reward_state') #nstep,nenv,n
            self.op_coef_mat = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='coef_mat') #nenv,nstep,nstep
            self.op_coef_mat_returns = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='coef_mat_returns') #nenv,nstep,nstep
            self.op_td = tf.placeholder(dtype=tf.float32, shape=[None, None], name='td') #nstep,nenv
            self.op_values = tf.placeholder(dtype=tf.float32, shape=[None, None], name='values') #nstep,nenv
            self.op_begin_mark = tf.placeholder(dtype=tf.float32, shape=[None, None], name='begin_mark') #nstep,nenv

            # Simulate GAE
            EPS = 1e-6
            #goal_velocity_params = tf.reshape(self.task.goal_velocity_params, (1, 1, self.task.n_dim))
            goal_velocity_params = self.task.goal_velocity_params
            coef = self.task.coef
            func = self.task.func
            logger.info(f'reward coef in ADVTASK is {coef}')
            logger.info(f'reward func in ADVTASK is {func}')
            if func == 'abs':
                logger.info('we are in abs func!')
                reward_params_gae = self.op_reward_ctrl - tf.reduce_sum(tf.abs(self.op_reward_state - goal_velocity_params)*coef, axis=2) + self.op_td #nstep,nenv
                reward_params = self.op_reward_ctrl - tf.reduce_sum(tf.abs(self.op_reward_state - goal_velocity_params)*coef, axis=2)
            elif func == 'linear':
                logger.info('we are in linear func!')
                reward_params_gae = self.op_reward_ctrl + tf.reduce_sum(tf.multiply(self.op_reward_state + EPS, goal_velocity_params*coef), axis=2) + self.op_td #nstep,nenv
                reward_params = self.op_reward_ctrl + tf.reduce_sum(tf.multiply(self.op_reward_state + EPS, goal_velocity_params*coef), axis=2)
            else:
                raise Exception(f'{FLAGS.task.reward} reward function is not available!')

            if self.atype == '1step':
                advantages_params = reward_params_gae  #r_t + \gamma V(s') - V(s)
            elif self.atype == 'gae':
                reward_params_gae = tf.transpose(reward_params_gae) #nenv,nstep
                advantages_params = tf.squeeze(tf.matmul(self.op_coef_mat, tf.expand_dims(reward_params_gae, 2)), [2]) #nenv,nstep,nstep * nenv,nstep,1 -> nenv,nstep
                advantages_params = tf.transpose(advantages_params) #nstep,nenv
            elif self.atype == 'ret':
                reward_params = tf.transpose(reward_params) #nenv,nstep
                advantages_params = tf.squeeze(tf.matmul(self.op_coef_mat_returns, tf.expand_dims(reward_params, 2)), [2]) #nenv,nstep,nstep * nenv,nstep,1 -> nenv,nstep
                advantages_params = tf.transpose(advantages_params) #nstep,nenv
            elif self.atype == 'adv':
                reward_params = tf.transpose(reward_params) #nenv,nstep  r_t
                advantages_params = tf.squeeze(tf.matmul(self.op_coef_mat_returns, tf.expand_dims(reward_params, 2)), [2]) #nenv,nstep,nstep * nenv,nstep,1 -> nenv,nstep
                advantages_params = tf.transpose(advantages_params) #nstep,nenv
                advantages_params = advantages_params - self.op_values  #\sum r_t - V(s)

            self.advantages_params = tf.reshape(advantages_params, (-1,)) #nstep*nenv
            self.advantages_params_pre = self.advantages_params
            self.adv_mean, self.adv_var = tf.nn.moments(tf.stop_gradient(self.advantages_params), axes=0)
            self.adv_std = tf.sqrt(self.adv_var)
            if self.advnormalize:
                self.advantages_params = (self.advantages_params - self.adv_mean) / tf.maximum(self.adv_std, 1e-8)
            self.op_advantages_params = self.advantages_params.reduce_mean()

            # reward function and returns
            if func == 'abs':
                reward_params = self.op_reward_ctrl - tf.reduce_sum(tf.abs(self.op_reward_state - goal_velocity_params)*coef, axis=2)
            elif func == 'linear':
                reward_params = self.op_reward_ctrl + tf.reduce_sum(tf.multiply(self.op_reward_state + EPS, goal_velocity_params*coef), axis=2)
            else:
                raise Exception(f'{FLAGS.task.reward} reward function is not available!')
            reward_params = tf.transpose(reward_params) #nenv,nstep
            returns_params = tf.squeeze(tf.matmul(self.op_coef_mat_returns, tf.expand_dims(reward_params, 2)), [2]) #nenv,nstep,nstep * nenv,nstep,1 -> nenv,nstep
            returns_params = tf.transpose(returns_params) #nstep,nenv
            #NOTE that NOT: self.returns_params = tf.reshape(returns_params, (-1,)) #nstep*nenv

            if self.meanret:
                self.returns_params = tf.reduce_sum(returns_params * self.op_begin_mark) / tf.reduce_sum(self.op_begin_mark * 0. + 1.)
            else:
                self.returns_params = tf.reduce_sum(returns_params * self.op_begin_mark) / tf.reduce_sum(self.op_begin_mark)
            returns_params_nozero = self.returns_params
            self.r_mean = self.adv_mean
            self.r_var = self.adv_var
            self.r_std = self.adv_std
            self.op_returns_params = self.returns_params.reduce_mean()

        # build loss from input placeholder
        self.op_policy_loss_warmup, self.op_policy_loss_warmup_quad, self.op_return_loss_warmup, self.op_policy_loss, self.op_return_loss= \
            self.build_loss(self.op_states, self.op_actions, self.advantages_params, self.returns_params)

        # compute jacobian and hessian part
        self.op_task_b, self.op_task_hvp = self.compute_task_jacobian(self.op_policy_loss_warmup, self.op_policy_loss_warmup_quad, self.op_tangents)

        # compute sync op and other gradients
        self.op_sync_old_warmup, self.op_flat_grad_policy_warmup, self.op_flat_grad_task_warmup = self.compute_grad(
                self.warmup_policy, self.old_warmup_policy, self.op_policy_loss_warmup, self.op_return_loss_warmup)
        self.op_sync_old, _, self.op_flat_grad_task_opt = self.compute_grad(
                self.policy, self.old_policy, self.op_policy_loss, self.op_return_loss)
        self.op_logpac_plot = self.warmup_policy(self.op_states).log_prob(self.op_actions).reduce_sum(axis=1).reduce_mean()

        # build sync all op
        assign_ops = []
        for old_v, new_v in zip(self.warmup_policy1.parameters(), self.warmup_policy.parameters()): assign_ops.append(tf.assign(old_v, new_v))
        for old_v, new_v in zip(self.warmup_policy2.parameters(), self.warmup_policy.parameters()): assign_ops.append(tf.assign(old_v, new_v))
        for old_v, new_v in zip(self.warmup_policy3.parameters(), self.warmup_policy.parameters()): assign_ops.append(tf.assign(old_v, new_v))
        self.op_sync_all = tf.group(*assign_ops)

    def compute_grad(self, policy, old_policy, policy_loss, return_loss): #loss={return_loss, policy_loss}, rollout data={read+hatpolicy,virtual+hatpolicy,real+optimalpolicy}
        policy_params = policy.parameters()
        old_params = old_policy.parameters()
        task_params = self.task.parameters()
        flat_grad_policy = nn.utils.parameters_to_vector(tf.gradients(policy_loss, policy_params))
        flat_grad_policy = tf.reshape(flat_grad_policy, (-1, 1))
        flat_grad_task = nn.utils.parameters_to_vector(tf.gradients(return_loss, task_params))
        sync_old = tf.group(*[tf.assign(old_v, new_v) for old_v, new_v in zip(old_params, policy_params)])
        return sync_old, flat_grad_policy, flat_grad_task

    def build_loss(self, states, actions, advantages_params, returns_params):
        # For warmup policy
        old_distribution: tf.distributions.Normal = self.old_warmup_policy(states)
        distribution: tf.distributions.Normal = self.warmup_policy(states)
        self.mean_kl = old_distribution.kl_divergence(distribution).reduce_sum(axis=1).reduce_mean()
        ratios: Tensor = (distribution.log_prob(actions) - old_distribution.log_prob(actions)) \
            .reduce_sum(axis=1).exp()
        logpac = distribution.log_prob(actions).reduce_sum(axis=1)

        #entropy = distribution.entropy().reduce_sum(axis=1).reduce_mean()
        #entropy_bonus = ent_coef * entropy
        policy_loss_warmup = logpac.mul(advantages_params).reduce_mean()  #policy gradient
        self.policy_loss_warmup = policy_loss_warmup
        return_loss_warmup = returns_params.reduce_mean()  #E_\tau [R(s_0)]
        self.logpac = logpac

        # For Computing hessian and quadratic grad
        logpac1 = self.warmup_policy1(states).log_prob(actions).reduce_sum(axis=1)
        logpac2 = self.warmup_policy2(states).log_prob(actions).reduce_sum(axis=1)
        logpac3 = self.warmup_policy3(states).log_prob(actions).reduce_sum(axis=1)
        self.logpac1 = logpac1
        self.logpac2 = logpac2
        self.logpac3 = logpac3
        policy_loss_warmup_quad = (logpac1 * logpac2 + logpac3).mul(tf.stop_gradient(advantages_params)).reduce_mean()
        self.quad_loss = (logpac * logpac + logpac).mul(tf.stop_gradient(advantages_params)).reduce_mean()

        # For optimal policy
        old_distribution: tf.distributions.Normal = self.old_policy(states)
        distribution: tf.distributions.Normal = self.policy(states)
        ratios: Tensor = (distribution.log_prob(actions) - old_distribution.log_prob(actions)) \
            .reduce_sum(axis=1).exp()
        logpac = distribution.log_prob(actions).reduce_sum(axis=1)
        policy_loss = logpac.mul(advantages_params).reduce_mean()  #policy gradient

        # For return loss
        return_loss = returns_params.reduce_mean()  #E_\tau [R(s_0)]

        return policy_loss_warmup, policy_loss_warmup_quad, return_loss_warmup, \
                policy_loss, return_loss

    def compute_task_jacobian(self, policy_loss, policy_loss_quad, tangents):
        # compute hvp
        params1 = self.warmup_policy1.parameters()
        params2 = self.warmup_policy2.parameters()
        params3 = self.warmup_policy3.parameters()
        self.op_task_hvp_Ax = nn.utils.quadgrad_vec_prod(policy_loss_quad, params1, params2, params3, tangents, AAx=False)
        task_hvp = nn.utils.quadgrad_vec_prod(policy_loss_quad, params1, params2, params3, tangents, AAx=True)
        if self.meanAAx:
            nparam = nn.utils.n_parameters_int(self.warmup_policy.parameters()).astype(np.float32)
            print ("meanAAx nparam:", nparam)
            print (type(nparam))
            task_hvp = task_hvp / nparam
        print ("task_hvp:", task_hvp)
        print ("task_hvp_Ax:", self.op_task_hvp_Ax)
        self.op_quad_hvp = nn.utils.hessian_vec_prod(self.quad_loss, self.warmup_policy.parameters(), tangents)
        self.op_hessian_hvp = nn.utils.hessian_vec_prod(policy_loss, self.warmup_policy.parameters(), tangents)
        self.op_hvp = nn.utils.hessian_vec_prod(self.mean_kl, self.warmup_policy.parameters(), tangents)

        # compute jacobian
        task_params = self.task.parameters()
        policy_params = self.warmup_policy.parameters()
        policy_gradient_flat = nn.utils.parameters_to_vector(tf.gradients(policy_loss, policy_params))
        self.pg_flat = policy_gradient_flat
        self.p_flat = nn.utils.parameters_to_vector(self.warmup_policy.parameters())
        print ("pg_flat:", self.pg_flat)
        print ("p_flat:", self.p_flat)

        task_jacobian = jacobian(policy_gradient_flat, task_params, use_pfor=False)
        print (task_jacobian.shape)
        task_jacobian = tf.reshape(task_jacobian, (-1, self.task.n_dim))
        if self.AAx:
            ATb = []
            for i in range(self.task.n_dim):
                ATb_i = nn.utils.quadgrad_vec_prod(policy_loss_quad, params1, params2, params3, task_jacobian[:,i], AAx=False)
                if self.meanAAx:
                    ATb_i = ATb_i / nn.utils.n_parameters_int(self.warmup_policy.parameters()).astype(np.float32)
                ATb.append(ATb_i)
            self.op_ATb = tf.stack(ATb, axis=0) #n_dim x |theta|
            print (self.op_ATb.shape)
            #exit(0)

        #############################################################
        jacobian_op = []
        task_gradients = nn.utils.parameters_to_vector(tf.gradients(policy_loss, task_params))
        self.task_gradients = task_gradients
        for i in range(self.task.n_dim):
            b = nn.utils.parameters_to_vector(tf.gradients(task_gradients[i], policy_params))
            jacobian_op.append(b)
        self.jacobian_op = tf.stack(jacobian_op, axis=0)
        print ("jacobian_op:", self.jacobian_op)
        print ("task_gradients:", self.task_gradients)
        print ("task_jacobian:", task_jacobian)
        #############################################################

        return task_jacobian, task_hvp

    def validate_advantage(self, advantages_pre, adv_mean, adv_std, feed_dict, name):
        advantages_post = (advantages_pre - adv_mean) / np.maximum(adv_std, 1e-8)
        advantages_params_post, advantages_params_pre, advp_mean, advp_std, returns_params, r_mean, r_std = tf.get_default_session().run(
                [self.advantages_params, self.advantages_params_pre, self.adv_mean, self.adv_std, self.returns_params, self.r_mean, self.r_std], feed_dict=feed_dict)
        print ('=====================================================%s=====================================================' % name)
        logger.info('Task goal_vel: %s', self.task.goal_velocity)
        print ('advp:', advp_mean, advp_std)
        print ('adv:', adv_mean, adv_std)
        #print ("returns_params:", returns_params.tolist())
        diff_pre = np.linalg.norm(advantages_pre - advantages_params_pre)
        diff_post = np.linalg.norm(advantages_post - advantages_params_post)
        print ("diff_pre: %f, diff_post: %f" % (diff_pre, diff_post))
        print ("advantage_pre[0:5]:", advantages_pre[0: 5])
        print ("advantage_params_pre[0:5]:", advantages_params_pre[0: 5])
        print ("advantage_post[0:5]:", advantages_post[0: 5])
        print ("advantage_params_post[0:5]:", advantages_params_post[0: 5])
        print ('====================================================================================================================')

    @nn.make_method(fetch='task_hvp')
    def get_task_hvp(self, states, tangents, actions, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, advantages_mean, advantages_std, values, begin_mark) -> List[np.ndarray]: pass

    @nn.make_method(fetch='task_hvp_Ax')
    def get_task_hvp_Ax(self, states, tangents, actions, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, advantages_mean, advantages_std, values, begin_mark) -> List[np.ndarray]: pass

    @nn.make_method(fetch='loss')
    def get_grad(self, states, actions, reward_ctrl, reward_state, coef_mat, coef_mat_returns, td, advantages_mean, advantages_std, values, begin_mark) -> List[np.ndarray]: pass

    @nn.make_method(fetch='sync_old')
    def sync_old(self) -> List[np.ndarray]: pass

    @nn.make_method(fetch='sync_old_warmup')
    def sync_old_warmup(self) -> List[np.ndarray]: pass

    @nn.make_method(fetch='sync_all')
    def sync_all(self) -> List[np.ndarray]: pass

    def get_grad_hatenv_hatpolicy(self, samples, advantages, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark):
        values = values.reshape((td.shape[0], td.shape[1]))
        adv_mean, adv_std = advantages.mean(), advantages.std()
        feed_dict = {
                self.op_states: samples.state, self.op_actions: samples.action,self.op_reward_ctrl: reward_ctrl, 
                self.op_reward_state: reward_state, self.op_coef_mat: coef_mat, self.op_coef_mat_returns: coef_mat_returns, self.op_td: td,
                self.op_advantages_mean: adv_mean, self.op_advantages_std: adv_std, self.op_values: values, self.op_begin_mark: begin_mark
        }
        b1 = self.get_grad(samples.state, samples.action, reward_ctrl, reward_state, coef_mat, coef_mat_returns, td, adv_mean, adv_std, values, begin_mark, fetch='task_b')#2768 x 2
        if self.AAx:
            ATb = self.get_grad(samples.state, samples.action, reward_ctrl, reward_state, coef_mat, coef_mat_returns, td, adv_mean, adv_std, values, begin_mark, fetch='ATb')#2 x 2768
        b4 = tf.get_default_session().run(self.jacobian_op, feed_dict=feed_dict) #2 x 2768
        # Solve Ax=b to get x = A^(-1)b
        def policy_hvp(x):
            return self.get_task_hvp(samples.state, x, samples.action, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, adv_mean, adv_std, values, begin_mark) + self.cg_damping * x
        def policy_hvp_Ax(x):
            return self.get_task_hvp_Ax(samples.state, x, samples.action, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, adv_mean, adv_std, values, begin_mark) + self.cg_damping * x
        policy_over_task_list = []
        self.policy_over_task_norm = []
        residual = []
        self.Ax_norm = []
        cg_mine = []
        for i in range(self.task.n_dim):
            if self.AAx:
                policy_over_task_i, residual_i, cg_mine_i = conj_grad(policy_hvp, ATb[i,:], n_iters=self.n_cg_iters, verbose=False)  #solve Ax=b
            else:
                policy_over_task_i, residual_i, cg_mine_i = conj_grad(policy_hvp, b[:,i], n_iters=self.n_cg_iters, verbose=False)  #solve Ax=b
            cg_mine.append(cg_mine_i)
            print ("Ax = ", policy_hvp_Ax(policy_over_task_i))
            print ("Ax norm = ", norm(policy_hvp_Ax(policy_over_task_i)))
            print ("xAx = ", policy_over_task_i.dot(policy_hvp_Ax(policy_over_task_i)))
            self.Ax_norm.append(norm(policy_hvp_Ax(policy_over_task_i)))
            #print ("residual = ", residual)
            residual.append(residual_i)
            policy_over_task_list.append(policy_over_task_i)
            self.policy_over_task_norm.append(norm(policy_over_task_i))
        self.info['cg_plot']['mine'].append(cg_mine)
        # gradient = -x
        policy_over_task = - np.array(policy_over_task_list)
        residual = np.mean(residual)
        return policy_over_task, residual

    def get_grad_optenv_hatpolicy(self, samples, advantages, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark):
        values = values.reshape((td.shape[0], td.shape[1]))
        adv_mean, adv_std = advantages.mean(), advantages.std()
        feed_dict = {
                self.op_states: samples.state, self.op_actions: samples.action,self.op_reward_ctrl: reward_ctrl, 
                self.op_reward_state: reward_state, self.op_coef_mat: coef_mat, self.op_coef_mat_returns: coef_mat_returns, self.op_td: td,
                self.op_advantages_mean: adv_mean, self.op_advantages_std: adv_std, self.op_values: values, self.op_begin_mark: begin_mark
        }
        self.validate_advantage(advantages, adv_mean, adv_std, feed_dict, 'flat_grad_policy')
        flat_grad_policy, flat_grad_task = self.get_grad(
                samples.state, samples.action, reward_ctrl, reward_state, coef_mat, coef_mat_returns, td, adv_mean, adv_std, values, begin_mark, 
                fetch='flat_grad_policy_warmup flat_grad_task_warmup')
        return flat_grad_policy, flat_grad_task

    def get_grad_optenv_optpolicy(self, samples, advantages, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark):
        values = values.reshape((td.shape[0], td.shape[1]))
        adv_mean, adv_std = advantages.mean(), advantages.std()
        feed_dict = {
                self.op_states: samples.state, self.op_actions: samples.action,self.op_reward_ctrl: reward_ctrl, 
                self.op_reward_state: reward_state, self.op_coef_mat: coef_mat, self.op_coef_mat_returns: coef_mat_returns, self.op_td: td,
                self.op_advantages_mean: adv_mean, self.op_advantages_std: adv_std, self.op_values: values, self.op_begin_mark: begin_mark
        }
        self.validate_advantage(advantages, adv_mean, adv_std, feed_dict, 'flat_grad_task_opt')
        flat_grad_task_opt = self.get_grad(
                samples.state, samples.action, reward_ctrl, reward_state, coef_mat, coef_mat_returns, td, adv_mean, adv_std, values, begin_mark, 
                fetch='flat_grad_task_opt')
        return flat_grad_task_opt

    def invalidate(self):
        for p in self.warmup_policy.parameters(): p.invalidate()
        for p in self.warmup_policy1.parameters(): p.invalidate()
        for p in self.warmup_policy2.parameters(): p.invalidate()
        for p in self.warmup_policy3.parameters(): p.invalidate()
        for p in self.policy.parameters(): p.invalidate()
        self.task.parameters().invalidate()

    def sync(self):
        self.sync_old(fetch='sync_old')
        self.sync_old_warmup(fetch='sync_old_warmup')
        self.sync_all(fetch='sync_all')

    def print_params_norm(self):
        pol_params, warm_params = tf.get_default_session().run([nn.utils.parameters_to_vector(self.policy.parameters()), nn.utils.parameters_to_vector(self.warmup_policy.parameters())])
        print ("In ADVTASK, pol_params_norm:", np.linalg.norm(pol_params), "warm_params_norm:", np.linalg.norm(warm_params))
        warm_params1, warm_params2, warm_params3 = tf.get_default_session().run([
            nn.utils.parameters_to_vector(self.warmup_policy1.parameters()),
            nn.utils.parameters_to_vector(self.warmup_policy2.parameters()),
            nn.utils.parameters_to_vector(self.warmup_policy3.parameters())
        ])
        print ("warm_params1_norm:", np.linalg.norm(warm_params1))
        print ("warm_params2_norm:", np.linalg.norm(warm_params2))
        print ("warm_params3_norm:", np.linalg.norm(warm_params3))

    def train(self, train_runner, collect_runner, warmup_collect_virt, warmup_collect_real, optimal_collect_real, returns_pre_warmup, val_losses_warmup, val_losses_slbo, train_losses_warmup, train_losses_slbo, surprisal, trpo_warmup, trpo_slbo, infofilename, extra_runners):
        #tf.get_default_session().run(tf.variables_initializer(self.warmup_policy.parameters()))
        self.sync()
        self.print_params_norm()
        self.invalidate()

        logger.info('--------------------------------------------------- Update Task Parameter ------------------------------------------------')
        self.task_num += 1

        sanitycheck = {}

        # collect virtual data
        logger.info('Rollout on virtual env, policy hat')
        train_runner.reset()
        data, ep_infos = train_runner.run(self.warmup_policy, self.nsample)
        advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark = train_runner.compute_advantage(self.warmup_vfn, data, self.task)

        # collect real data
        #logger.info('Rollout on real env, policy hat')
        #collect_runner.reset()
        #data, ep_infos = collect_runner.run(self.warmup_policy, self.nsample)
        #advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark = collect_runner.compute_advantage(self.warmup_vfn, data, self.task)
        logp = self.warmup_policy(data.state).log_prob(data.action).reduce_sum(axis=1).reduce_mean()
        logp = tf.get_default_session().run(logp)
        print ("state_mean:", np.mean(data.state))
        print ("action_mean:", np.mean(data.action))
        print ("warmup_logpac_mean in ADVTASK:", logp)
        #ep_infos, data, advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark = warmup_collect_virt
        policy_over_task, cg_residual = self.get_grad_hatenv_hatpolicy(data, advantages, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark)
        virtual_returns_post_warmup = np.mean([info['return'] for info in ep_infos])
        print ("policy_over_task.shape:", policy_over_task.shape)
        #exit(0)

        logger.info('Rollout on real env, policy hat')
        collect_runner.reset()
        data, ep_infos = collect_runner.run(self.warmup_policy, self.nsample)
        advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark = collect_runner.compute_advantage(self.warmup_vfn, data, self.task)
        #ep_infos, data, advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark = warmup_collect_real
        returns_post_warmup = np.mean([info['return'] for info in ep_infos])
        flat_grad_policy, flat_grad_task = self.get_grad_optenv_hatpolicy(data, advantages, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark)
        print ("flat_grad_policy.shape:", flat_grad_policy.shape)
        print ("flat_grad_task.shape:", flat_grad_task.shape)


        logger.info('Rollout on real env, policy star')
        # collect data with optimal policy
        collect_runner.reset()
        data, ep_infos = collect_runner.run(self.policy, self.nsample)
        advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark = collect_runner.compute_advantage(self.vfn, data, self.task)
        #ep_infos, data, advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark = optimal_collect_real
        returns_post_slbo = np.mean([info['return'] for info in ep_infos])
        flat_grad_task_opt = self.get_grad_optenv_optpolicy(data, advantages, values, td, coef_mat, coef_mat_returns, reward_ctrl, reward_state, begin_mark)
        print ("flat_grad_task_opt.shape:", flat_grad_task_opt.shape)

        ### Compute
        matmul = np.reshape(np.matmul(policy_over_task, flat_grad_policy), (self.task.n_dim,))
        final_grad = (flat_grad_task_opt - flat_grad_task) - matmul

        #max L over psi -> get gradient ascent
        goal_velocity = self.task.goal_velocity
        task_params_before = tf.get_default_session().run(self.task.parameters())
        task_params_after = tf.get_default_session().run(self.task.parameters())
        # Perform line search
        #self.task.set_parameters(task_params_after)
        goal_velocity_after = tf.get_default_session().run(self.task.goal_velocity_params)
        print ("final_grad:", final_grad.shape)
        print ("before(delete):", task_params_before.shape)
        print ("after(delete):", task_params_after.shape)

        _, ep_infos = train_runner.run(self.policy, self.nsample)
        virtual_returns_post_slbo = np.mean([info['return'] for info in ep_infos])
        diff = returns_post_slbo - returns_post_warmup
        virtual_diff = virtual_returns_post_slbo - virtual_returns_post_warmup

        warmup_params = tf.get_default_session().run(nn.utils.parameters_to_vector(self.warmup_policy.parameters()))
        policy_params = tf.get_default_session().run(nn.utils.parameters_to_vector(self.policy.parameters()))
        warmup_policy_norm = np.linalg.norm(warmup_params)
        policy_norm = np.linalg.norm(policy_params)
        print ("alpha and beta:", self.alpha, self.beta)
        print ('warmup_policy_norm:', warmup_policy_norm)
        print ('policy_norm:', policy_norm)
        print ("logpac_norm:", self.logpac_norm)
        print ("pg_norm:", self.pg_norm)
        print ("Ax_norm:", self.Ax_norm)
        print ("b_norm:", self.b_norm)
        print ("cg_residual:", cg_residual)
        x_norm = self.policy_over_task_norm
        print ("x_norm (policy_over_task_norm):", self.policy_over_task_norm)
        flat_grad_policy_norm = np.linalg.norm(flat_grad_policy)
        print ("flat_grad_policy_norm:", np.linalg.norm(flat_grad_policy))
        print ("flat_grad_task_opt:", flat_grad_task_opt)
        print ("flat_grad_task:", flat_grad_task)
        print ("minus:", flat_grad_task_opt - flat_grad_task)
        print ("matmul:", matmul)
        print ("final_grad:", final_grad)
        print (f"loss and adv: {self.policy_loss_value}, {self.quad_loss_value}, {self.return_loss_value}, {self.return_mean_value}, {self.adv_mean_value}")
        print (f'task_params: {task_params_before} -> {task_params_after}')
        print (f'goal_vel: {goal_velocity} -> {goal_velocity_after}')
        print (f'returns_pre_warmup={returns_pre_warmup}, returns_post_warmup={returns_post_warmup}, returns_post_slbo={returns_post_slbo}, real_returns_diff={diff}')
        print (f'virtual_returns_post_warmup={virtual_returns_post_warmup}, virtual_returns_post_slbo={virtual_returns_post_slbo}, virtual_diff={virtual_diff}')
        print (f'val_losses_warmup={np.mean(val_losses_warmup)}, val_losses_slbo={np.mean(val_losses_slbo)}')
        print (f'train_losses_warmup={np.mean(train_losses_warmup)}, train_losses_slbo={np.mean(train_losses_slbo)}')
        print (f'#val_losses_warmup={len(val_losses_warmup)}, #val_losses_slbo={len(val_losses_slbo)}')
        print (f'surprisal={surprisal}')
        trpo_slbo= np.array(trpo_slbo)
        print ('=======================')
        print (trpo_slbo.shape)
        trpo_plot = np.mean(trpo_slbo, 0).tolist()
        print ("trpo_plot:", trpo_plot)

        self.info['goal_velocity'].append(goal_velocity)
        self.info['matmul'].append(matmul)
        self.info['flat_grad_task'].append(flat_grad_task)
        self.info['flat_grad_task_opt'].append(flat_grad_task_opt)
        self.info['final_grad'].append(final_grad)
        self.info['task_params_before'].append(task_params_before)
        self.info['task_params_after'].append(task_params_after)
        self.info['returns_pre_warmup'].append(returns_pre_warmup)
        self.info['returns_post_warmup'].append(returns_post_warmup)
        self.info['returns_post_slbo'].append(returns_post_slbo)
        self.info['diff'].append(diff)
        self.info['virtual_returns_post_warmup'].append(virtual_returns_post_warmup)
        self.info['virtual_returns_post_slbo'].append(virtual_returns_post_slbo)
        self.info['virtual_diff'].append(virtual_diff)
        self.info['cg_residual'].append(cg_residual)
        self.info['val_losses_warmup'].append(np.mean(val_losses_warmup))
        self.info['val_losses_slbo'].append(np.mean(val_losses_slbo))
        self.info['train_losses_warmup'].append(np.mean(train_losses_warmup))
        self.info['train_losses_slbo'].append(np.mean(train_losses_slbo))
        self.info['surprisal'].append(surprisal)
        self.info['warmup_policy_norm'].append(warmup_policy_norm)
        self.info['policy_norm'].append(policy_norm)
        self.info['logpac_norm'].append(self.logpac_norm)
        self.info['pg_norm'].append(self.pg_norm)
        self.info['Ax_norm'].append(self.Ax_norm)
        self.info['b_norm'].append(self.b_norm)
        self.info['x_norm'].append(x_norm)
        self.info['flat_grad_policy_norm'].append(flat_grad_policy_norm)
        self.info['policy_loss_value'].append(self.policy_loss_value)
        self.info['quad_loss_value'].append(self.quad_loss_value)
        self.info['return_loss_value'].append(self.return_loss_value)
        self.info['return_mean_value'].append(self.return_mean_value)
        self.info['adv_mean_value'].append(self.adv_mean_value)
        self.info['sanitycheck'].append(sanitycheck)
        self.invalidate()
        norm_g = np.linalg.norm(final_grad)
        print (f"norm_g = {norm_g}")
        print (f"task_num = {self.task_num}")
        return task_params_before, final_grad, self.info
