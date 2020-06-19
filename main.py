import sys
sys.path = ['./rllab/'] + sys.path
print (sys.path)
import pickle
import os,time
from collections import deque
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger
from slbo.utils.average_meter import AverageMeter
from slbo.utils.flags import FLAGS
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.utils.OU_noise import OUNoise
from slbo.utils.normalizer import Normalizers
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.runner import Runner
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.envs.virtual_env import VirtualEnv
from slbo.dynamics_model import DynamicsModel
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.partial_envs import make_env, make_task
from slbo.loss.multi_step_loss import MultiStepLoss
from slbo.algos.TRPO import TRPO
from slbo.algos.ADVTASK import ADVTASK
from slbo.utils.tf_utils import initialize_uninitialized
import click
from gym.wrappers.monitor import Monitor
import gym
import scipy.misc
import scipy.ndimage
def render(env_, policy=None):
    logger.info('start render video...')
    observation = env_.reset()
    imgs = []
    return_ = 0.
    cnt_ = 0
    obs = []
    for t in range(200):
        cnt_ += 1
        observation = observation.reshape(1, -1)
        obs.append(observation)
        if policy is not None:
            action = policy.get_actions(observation)
            observation, reward, done, info = env_.step(action[0])
            if done: break
            return_ += reward
        else:
            action = env_.action_space.sample()
            observation, reward, done, info = env_.step(action)
            if done: break
            return_ += reward
    logger.info (f"render {cnt_} steps, return = {return_:.6f}")
    res = {'obs': obs, 'return': return_}
    return res

def eval_rollout(runner, p, des):
    logger.info(des)
    runner.reset()
    data, ep_infos = runner.run(p, FLAGS.plan.n_trpo_samples)
    logp = p(data.state).log_prob(data.action).reduce_sum(axis=1).reduce_mean()
    logp = tf.get_default_session().run(logp)
    print ("state_mean:", np.mean(data.state))
    print ("action_mean:", np.mean(data.action))
    print ("warmup_logpac_mean:", logp)

def testeval(policy, runner):
    runner.reset()
    _, ep_infos = runner.run(policy, FLAGS.rollout.n_test_samples)
    returns = [info['return'] for info in ep_infos]
    returns = np.mean(returns)
    return returns

def evaluate(settings, tag):
    res = {}
    for runner, policy, name in settings:
        runner.reset()
        _, ep_infos = runner.run(policy, FLAGS.rollout.n_test_samples)
        returns = np.array([ep_info['return'] for ep_info in ep_infos])
        res[name] = np.mean(returns)
        logger.info('Tag = %s, Reward on %s (%d episodes): mean = %.6f, std = %.6f', tag, name,
                    len(returns), np.mean(returns), np.std(returns))
    return res['Real Env'], res['Virt Env']


def add_multi_step(src: Dataset, dst: Dataset):
    n_envs = 1
    dst.extend(src[:-n_envs])

    ending = src[-n_envs:].copy()
    ending.timeout = True
    dst.extend(ending)


def make_real_runner(n_envs, task_config=None):
    from slbo.envs.batched_env import BatchedEnv
    batched_env = BatchedEnv([make_env(FLAGS.env.id, task_config=task_config) for _ in range(n_envs)])
    return Runner(batched_env, rescale_action=True, **FLAGS.runner.as_dict())


@click.command()
@click.option('--setting', default='default')
@click.option('--adv', default=1)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True, default=False)
@click.option('--taskname', default='Ant2D')
@click.option('--verbose', is_flag=True, default=False)
@click.option('--test', is_flag=True, default=False)
@click.option('--warmupent', default=0.005)
@click.option('--alpha', default=1.0)
@click.option('--beta', default=1.0)
@click.option('--snapshot', default=1)
@click.option('--testadv', default=0)
@click.option('--seed', default=1)
@click.option('--nsample', default=10000)
@click.option('--fixedvel', default=None)
@click.option('--initnslbo', default=20)
@click.option('--nslbo', default=3)
@click.option('--warmniter', default=40)
@click.option('--slboniter', default=20)
@click.option('--piter', default=20)
@click.option('--miter', default=100)
@click.option('--atype', default='gae') # gae, 1step, ret, adv
@click.option('--video', is_flag=True, default=False)
@click.option('--maxstep', default=1)
@click.option('--genadvstrategy', default=None)
@click.option('--inittask', default='none')
@click.option('--decay', default='joint')
@click.option('--testgiven', default=None)
@click.option('--testnum', default=1)
@click.option('--testparam', default='')
def main(setting, adv, gpu, debug, taskname, verbose, test, warmupent, alpha, beta, snapshot, testadv, seed, nsample, fixedvel, initnslbo, nslbo, warmniter, slboniter, piter, miter, atype, video, maxstep, genadvstrategy, inittask, decay, testgiven, testnum, testparam):
    print ('warmupent:', warmupent)
    print ("seed:", seed)
    setting = os.path.join('./data/', setting)
    #FLAGS.run_id = setting
    FLAGS.rollout.n_train_samples = 10000
    FLAGS.rollout.n_dev_samples = 10000
    FLAGS.rollout.n_test_samples = 10000
    FLAGS.plan.n_trpo_samples = 10000
    if taskname == 'HC':
        FLAGS.env.id = 'HalfCheetahTask-v2'
    elif taskname == 'HC2D':
        FLAGS.env.id = 'HalfCheetah2D-v2'
    elif taskname == 'HClinearstate':
        FLAGS.env.id = 'HalfCheetahLinearState-v2'
    elif taskname == 'HCgoalstate':
        FLAGS.env.id = 'HalfCheetahGoalState-v2'
    elif taskname == 'Hopper2D':
        FLAGS.env.id = 'Hopper2D-v2'
    elif taskname == 'Walker2D':
        FLAGS.env.id = 'Walker2D-v2'
    elif taskname == 'Ant3D':
        FLAGS.env.id = 'Ant3DTask-v2'
    elif taskname == 'Ant2D':
        FLAGS.env.id = 'Ant2DTask-v2'
    else:
        raise Exception(f'Unsupported taskname: {taskname}')
    if not os.path.isdir(setting):
        os.makedirs(setting)
    if not test:
        filename = f'res_{taskname}_adv{adv}.txt'
        infofilename = f'res_{taskname}_adv{adv}.npy'
        filename = setting+'/'+filename
        infofilename = setting+'/'+infofilename
        fout = open(filename, 'w')
    else:
        maxstep = 100
        logger.info(f'fixedvel={fixedvel}')
        if testadv:
            logger.info('Test with adversarial generated tasks!')
            logger.info(f'testadv=1, maxstep={maxstep}, using model revert!')
        else:
            logger.info('We still do not consider this senario: test with random tasks')
    print ('adv=', adv)
    FLAGS.seed = seed
    FLAGS.set_seed()
    FLAGS.freeze()
    print ("FLAGS.log_dir:", FLAGS.log_dir)
    if test:
        model_load = f'{FLAGS.log_dir}/{taskname}-stage-{snapshot}.npy'
    else:
        model_load = None
    print ("model_load:", model_load)

    task = make_task(FLAGS.env.id)
    env = make_env(FLAGS.env.id, task_config=task)
    dim_state = int(np.prod(env.observation_space.shape))
    dim_action = int(np.prod(env.action_space.shape))

    env.verify()

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    normalizers_copy = Normalizers(dim_action=dim_action, dim_state=dim_state)
    normalizers_parameters = normalizers.parameters(trainable=False, non_trainable=True)
    normalizers_copy_parameters = normalizers_copy.parameters(trainable=False, non_trainable=True)
    copy_normalizers = tf.group(*[tf.assign(w_v, p_v) for w_v, p_v in zip(normalizers_copy_parameters, normalizers_parameters)])
    revert_normalizers = tf.group(*[tf.assign(w_v, p_v) for w_v, p_v in zip(normalizers_parameters, normalizers_copy_parameters)])

    dtype = gen_dtype(env, 'state action next_state reward done timeout')
    train_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
    dev_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
    task_train_sets = [Dataset(dtype, FLAGS.rollout.max_buf_size) for i in range(100)]
    task_dev_sets = [Dataset(dtype, FLAGS.rollout.max_buf_size) for i in range(100)]

    print ("state and action dim:", dim_state, dim_action)
    policy = GaussianMLPPolicy(dim_state, dim_action, normalizer=normalizers.state, **FLAGS.policy.as_dict())
    warmup_policy = GaussianMLPPolicy(dim_state, dim_action, normalizer=normalizers.state, **FLAGS.policy.as_dict())
    print (policy.parameters())
    print (warmup_policy.parameters())
    sync_warmup_policy = tf.group(*[tf.assign(w_v, p_v) for w_v, p_v in zip(warmup_policy.parameters(), policy.parameters())])
    # batched noises
    noise = OUNoise(env.action_space, theta=FLAGS.OUNoise.theta, sigma=FLAGS.OUNoise.sigma, shape=(1, dim_action))
    vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    warmup_vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    sync_warmup_vfn = tf.group(*[tf.assign(w_v, p_v) for w_v, p_v in zip(warmup_vfn.parameters(), vfn.parameters())])

    model = DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)
    lazy_model = DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)
    warmup_model = DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)
    sync_warmup_model = tf.group(*[tf.assign(w_v, p_v) for w_v, p_v in zip(warmup_model.parameters(), model.parameters())])
    shadow_models = [DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes) for n in range(FLAGS.warmup.n_shadow_models)]
    sync_model_from_lazymodel = tf.group(*[tf.assign(w_v, p_v) for w_v, p_v in zip(model.parameters(), lazy_model.parameters())])
    sync_model_to_lazymodel = tf.group(*[tf.assign(w_v, p_v) for w_v, p_v in zip(lazy_model.parameters(), model.parameters())])

    virt_env = VirtualEnv(model, make_env(FLAGS.env.id, task_config=task), FLAGS.plan.n_envs, opt_model=FLAGS.slbo.opt_model)
    virt_runner = Runner(virt_env, **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps})
    virt_env_copy = VirtualEnv(model, make_env(FLAGS.env.id, task_config=task), nsample//FLAGS.plan.max_steps, opt_model=FLAGS.slbo.opt_model)
    virt_runner_copy = Runner(virt_env_copy, **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps})
    extra_runners = {}
    for sam in [1000, 2000, 4000, 8000, 10000, 16000]:
        extra_runners[f'train{sam}']= Runner(VirtualEnv(model, make_env(FLAGS.env.id, task_config=task), sam//FLAGS.plan.max_steps, opt_model=FLAGS.slbo.opt_model), **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps})
        extra_runners[f'collect{sam}'] = make_real_runner(sam//FLAGS.plan.max_steps, task_config=task)

    warmup_virt_env = VirtualEnv(warmup_model, make_env(FLAGS.env.id, task_config=task), FLAGS.plan.n_envs, opt_model=FLAGS.slbo.opt_model)
    warmup_virt_runner = Runner(warmup_virt_env, **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps})

    logger.info('FLAGS.plan.n_envs=%d' % FLAGS.plan.n_envs)
    shadow_envs = [VirtualEnv(shadow_model, make_env(FLAGS.env.id, task_config=task), FLAGS.plan.n_envs, opt_model=FLAGS.slbo.opt_model) for shadow_model in shadow_models]
    shadow_runners = [Runner(shadow_env, **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps}) for shadow_env in shadow_envs]

    criterion_map = {
        'L1': nn.L1Loss(),
        'L2': nn.L2Loss(),
        'MSE': nn.MSELoss(),
    }
    criterion = criterion_map[FLAGS.model.loss]
    loss_mod = MultiStepLoss(model, normalizers, dim_state, dim_action, criterion, FLAGS.model.multi_step)
    loss_mod.build_backward(FLAGS.model.lr, FLAGS.model.weight_decay)
    shadow_loss_mods = [MultiStepLoss(shadow_model, normalizers, dim_state, dim_action, criterion, FLAGS.model.multi_step) for shadow_model in shadow_models]
    for shadow_loss_mod in shadow_loss_mods:
        shadow_loss_mod.build_backward(FLAGS.model.lr, FLAGS.model.weight_decay)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.as_dict())
    advtask = ADVTASK(dim_state, dim_action, policy, vfn, warmup_policy, warmup_vfn, task, alpha=alpha, beta=beta, nsample=nsample, atype=atype)
    tf.get_default_session().run(tf.global_variables_initializer())

    print ("norm params:", normalizers_parameters)
    print ("norm_copy params:", normalizers_copy_parameters)
    norm_before = tf.get_default_session().run(normalizers_parameters)
    print ("norm_before:", norm_before)

    assert FLAGS.algorithm != 'MF', "don't support model free for now"

    print (f"n_envs for task: {nsample}//{FLAGS.plan.max_steps}={nsample//FLAGS.plan.max_steps}")

    runners = {
        'test': make_real_runner(FLAGS.plan.n_envs, task_config=task),
        'collect': make_real_runner(FLAGS.plan.n_envs, task_config=task), #1
        'collect_copy': make_real_runner(nsample//FLAGS.plan.max_steps, task_config=task), #1
        'dev': make_real_runner(FLAGS.plan.n_envs, task_config=task),
        'train': make_real_runner(FLAGS.plan.n_envs, task_config=task) if FLAGS.algorithm == 'MF' else virt_runner,
        'train_copy': make_real_runner(nsample//FLAGS.plan.max_steps, task_config=task) if FLAGS.algorithm == 'MF' else virt_runner_copy,
        'warmup_train': make_real_runner(FLAGS.plan.n_envs, task_config=task) if FLAGS.algorithm == 'MF' else warmup_virt_runner,
    }
    for name, runner in extra_runners.items():
        runners[name] = runner
        print ("runner name is ", name)
    settings = [(runners['test'], policy, 'Real Env'), (runners['train'], policy, 'Virt Env')]
    for (i, runner) in enumerate(shadow_runners):
        settings.append((runner, policy, f'Shadow Env-{i}'))

    saver = nn.ModuleDict({'policy': policy, 'model': model, 'vfn': vfn, 'normalizers': normalizers})  #, 'loss_mod': loss_mod})
    print(saver)

    max_ent_coef = FLAGS.TRPO.ent_coef

    skip_metrics = []
    TASK_NUM = 0

    if test:
        verbose = True
    else:
        task.init()
        print (f"task.params_={task.params_}, task.init_goal_vel={task.goal_velocity}")

    if test:
        ITERS = testnum + 1
        warmup_n_iters = warmniter
        warmup_n_policy_iters = piter
        warmup_n_model_iters = miter
        slbo_n_iters = slboniter
        slbo_n_policy_iters = piter
        slbo_n_model_iters = miter
    else:
        ITERS = FLAGS.task.n_iters
        warmup_n_iters = warmniter
        warmup_n_policy_iters = piter
        warmup_n_model_iters = miter
        slbo_n_iters = slboniter
        slbo_n_policy_iters = piter
        slbo_n_model_iters = miter
    print (f"Total Iters = {ITERS}")
    alltaskres = []
    generated_adversarial_task = []
    init_generator = False
    logger.info(f'inittask:{inittask}')
    if not test:
        if inittask == 'none':
            pass
        elif not (os.path.exists(f'./{inittask}/{taskname}.trainset.task0.slbo0.pkl') and os.path.exists(f'./{inittask}/{taskname}.task0.saver.npy')):
            init_generator = True
        else:
            logger.info('Load the first task dataset!')
            for i in range(20):
                if not os.path.exists(f'./{inittask}/{taskname}.trainset.task0.slbo{i}.pkl'): continue
                traindata = pickle.load(open(f'./{inittask}/{taskname}.trainset.task0.slbo{i}.pkl', 'rb'))
                add_multi_step(traindata, train_set)
                add_multi_step(traindata, task_train_sets[0])
                logger.info(f'load trainset-{i} {len(traindata)}')

            for i in range(20):
                if not os.path.exists(f'./{inittask}/{taskname}.devset.task0.slbo{i}.pkl'): continue
                devdata = pickle.load(open(f'./{inittask}/{taskname}.devset.task0.slbo{i}.pkl', 'rb'))
                add_multi_step(devdata, task_dev_sets[0])
                logger.info(f'load devset-{i} {len(devdata)}')

            logger.info('Load the first task saver!')
            saver.load_state_dict(np.load(f'./{inittask}/{taskname}.task0.saver.npy', allow_pickle=True)[()])

            logger.info('Update all copies! (lazymodel, normalizers_copy)')
            tf.get_default_session().run(sync_model_to_lazymodel)
            tf.get_default_session().run(copy_normalizers)
            logger.info('Loaded normalizers:')
            load_norm = tf.get_default_session().run(normalizers_parameters)
            logger.info(load_norm)
            TASK_NUM = 1
            ########################## debug #########################
            #for task_idx in range(TASK_NUM):
            #    total_loss = []
            #    for scan in range(100):
            #        samples = task_train_sets[task_idx].sample_multi_step(FLAGS.model.dev_batch_size, 1, FLAGS.model.multi_step)
            #        loss_i = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
            #        total_loss.append(loss_i.mean())
            #    total_loss = np.mean(total_loss)
            #    print ('loaded model train loss:', total_loss)

            #for task_idx in range(TASK_NUM):
            #    total_loss = []
            #    for scan in range(100):
            #        samples = task_dev_sets[task_idx].sample_multi_step(FLAGS.model.dev_batch_size, 1, FLAGS.model.multi_step)
            #        loss_i = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
            #        total_loss.append(loss_i.mean())
            #    total_loss = np.mean(total_loss)
            #    print ('loaded model val loss:', total_loss)
            ##exit(0)
            ########################## debug #########################
    else:
        test_summary = {
            'task':[],
            'random':[],
            'warmup':[],
            'warmupprocess':[],
            'slbo':[],
        }
        logger.info('Testing mode!')
        train_tasknum = snapshot + 1
        test_tasknum = testnum
        logger.info(f'train_tasknum = {train_tasknum}, test_tasknum = {test_tasknum}')
        assert(testgiven is not None)
        if 'noent' in testparam: warmupent = 0.
        have_data = False

        task_generator = 'fixed' # random or fixed
        if testgiven[-4:] == '.pkl':
            f = testgiven
            logger.info(f'Load all tasks from {f}!')
            task.fixed_velocities = pickle.load(open(f, 'rb'))
            logger.info(f"Test on task")
            logger.info(task.fixed_velocities)
            logger.info(f"Task number: {np.array(task.fixed_velocities).shape}")
        else:
            f = f'{testgiven}/all_task_parameter.pkl'
            gen_adv_task = pickle.load(open(f, 'rb'))
            logger.info(f'Load all adversarial task from {f}!')
            task.fixed_velocities = gen_adv_task[train_tasknum: train_tasknum + test_tasknum]
            logger.info(f"Test random method on task {train_tasknum}~{train_tasknum+test_tasknum}:")
            logger.info(task.fixed_velocities)
            logger.info(f"Task number: {np.array(task.fixed_velocities).shape}")

        def load_data_during_test():
            if inittask != 'none':
                logger.info('Load the first task dataset!')
                for i in range(20):
                    if not os.path.exists(f'./{inittask}/{taskname}.trainset.task0.slbo{i}.pkl'): continue
                    traindata = pickle.load(open(f'./{inittask}/{taskname}.trainset.task0.slbo{i}.pkl', 'rb'))
                    add_multi_step(traindata, train_set)
                    add_multi_step(traindata, task_train_sets[0])
                    logger.info(f'load task0 trainset{i} size={len(traindata)}')
                    have_data = True

                for i in range(20):
                    if not os.path.exists(f'./{inittask}/{taskname}.devset.task0.slbo{i}.pkl'): continue
                    devdata = pickle.load(open(f'./{inittask}/{taskname}.devset.task0.slbo{i}.pkl', 'rb'))
                    add_multi_step(devdata, task_dev_sets[0])
                    logger.info(f'load task0 devset{i} size={len(devdata)}')
                    have_data = True

            logger.info(f'Load all task dataset from {setting}!')
            for t in range(0,train_tasknum):
                for i in range(20):
                    if not os.path.exists(f'./{setting}/{taskname}.trainset.task{t}.slbo{i}.pkl'): continue
                    traindata = pickle.load(open(f'./{setting}/{taskname}.trainset.task{t}.slbo{i}.pkl', 'rb'))
                    add_multi_step(traindata, train_set)
                    add_multi_step(traindata, task_train_sets[t])
                    logger.info(f'load task{t} trainset{i} size={len(traindata)}')
                    if not os.path.exists(f'./{setting}/{taskname}.devset.task{t}.slbo{i}.pkl'): continue
                    devdata = pickle.load(open(f'./{setting}/{taskname}.devset.task{t}.slbo{i}.pkl', 'rb'))
                    add_multi_step(devdata, task_dev_sets[t])
                    logger.info(f'load task{t} devset{i} size={len(devdata)}')
                    have_data = True
        load_data_during_test()

        logger.info(f'Load the task{snapshot} saver!')
        saver.load_state_dict(np.load(f'./{setting}/{taskname}.task{snapshot}.saver.npy', allow_pickle=True)[()])

        logger.info('Update all copies! (lazymodel, normalizers_copy)')
        tf.get_default_session().run(sync_model_to_lazymodel)
        tf.get_default_session().run(copy_normalizers)
        logger.info('Loaded normalizers:')
        load_norm = tf.get_default_session().run(normalizers_parameters)
        logger.info(load_norm)

        TASK_NUM = train_tasknum
        TEST_TASK_NUM = 0
        ########################## debug #########################
        #if have_data:
        #    for task_idx in range(TASK_NUM):
        #        total_loss = []
        #        for scan in range(100):
        #            samples = task_train_sets[task_idx].sample_multi_step(FLAGS.model.dev_batch_size, 1, FLAGS.model.multi_step)
        #            loss_i = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
        #            total_loss.append(loss_i.mean())
        #        total_loss = np.mean(total_loss)
        #        print ('loaded model train loss:', total_loss)

        #    for task_idx in range(TASK_NUM):
        #        total_loss = []
        #        for scan in range(100):
        #            samples = task_dev_sets[task_idx].sample_multi_step(FLAGS.model.dev_batch_size, 1, FLAGS.model.multi_step)
        #            loss_i = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
        #            total_loss.append(loss_i.mean())
        #        total_loss = np.mean(total_loss)
        #        print ('loaded model val loss:', total_loss)
        ##exit(0)
        ######################### debug #########################
        slbo_n_stages = nslbo
        print (f"each task will do nslbo = {nslbo}")
        for param in model.parameters():
            param.invalidate()

    all_task_parameter = []
    while (not test and TASK_NUM < ITERS) or (test and TEST_TASK_NUM < ITERS):
        # first task or maxstep, update the model. Otherwise, revert the model
        logger.info('Sync model from lazymodel')
        tf.get_default_session().run(sync_model_from_lazymodel)
        taskres = {}
        if 'goal_velocity' not in taskres.keys():
            taskres['goal_velocity'] = []
        if not test and inittask == 'none':
            slbo_n_stages = nslbo
        elif not test and TASK_NUM == 0:
            slbo_n_stages = initnslbo
        elif not test and TASK_NUM > 0:
            slbo_n_stages = nslbo

        time_start = time.time()
        trpo_warmup = []
        trpo_slbo = []
        surprisal = []
        train_losses_warmup = deque(maxlen=warmup_n_model_iters // FLAGS.model.validation_freq)
        train_losses_slbo = deque(maxlen=slbo_n_model_iters // FLAGS.model.validation_freq)
        val_losses_warmup = deque(maxlen=warmup_n_model_iters // FLAGS.model.validation_freq)
        val_losses_slbo = deque(maxlen=slbo_n_model_iters // FLAGS.model.validation_freq)
        # NOTE: For each test task, we should reset model to the loaded one, and randomly initialize policy and vfn
        #if test:
        #    saver.load_state_dict(np.load(model_load, allow_pickle=True)[()])
        #    logger.warning('Load model from %s', model_load)
        if test:
            logger.info("################################################## TESTING TASK %d ################################################", TEST_TASK_NUM)
            logger.info(f'TEST_TASK_NUM={TEST_TASK_NUM}, TASK_NUM={TASK_NUM}')
            logger.warning('Revert model and normalizers')
            tf.get_default_session().run(sync_model_from_lazymodel)
            tf.get_default_session().run(revert_normalizers)
        else:
            logger.info("################################################## TRAINING TASK %d ################################################", TASK_NUM)
        if test:
            test_returns = []
            test_summary['warmupprocess'].append([])
            test_summary['slbo'].append([])
        if not test: #and FLAGS.task.method == 'random':
            if inittask != 'none' and TASK_NUM == 1:
                if 'HClinearstate' in taskname:
                    task.init([0.2] * task.n_params)
                else:
                    task.init([0.] * task.n_params)
            else:
                if TASK_NUM > 0: #fix the 1st tasks during training
                    if adv == 0:
                        task.random_sample('uniform')
                    elif adv == 2:
                        task.random_sample('normal')
                    elif adv == 1:
                        if TASK_NUM == 1 and inittask != 'none':
                            task.random_sample()
                    print (f"task.params_={task.params_}, task.init_goal_vel={task.goal_velocity}")
                    task.sample(adv=True)
            logger.info('Task Sampled: %s', task.goal_velocity)
            taskres['goal_velocity'].append(task.goal_velocity)
            all_task_parameter.append(task.goal_velocity)
            print (f"task.params_={task.params_}, task.init_goal_vel={task.goal_velocity}")
        if test:
            if task_generator == 'fixed':
                task.goal_velocity = task.fixed_velocities[TEST_TASK_NUM] #TODO
                logger.info('Task Fixed: %s', task.goal_velocity)
            if task_generator == 'random':
                task.sample(adv=False)  #sample randomly
                logger.info('Task Sampled: %s', task.goal_velocity)
            if task_generator == 'adv':
                task.sample(adv=True)  #sample adversarially
                logger.info('Task Sampled: %s', task.goal_velocity)
                generated_adversarial_task.append(task.goal_velocity)
                logger.info('Tasks dump!')
            assert (task_generator == 'fixed')
            test_summary['task'].append(task.goal_velocity)

        if FLAGS.task.reset_policy:
            # NOTE: reset policy and valuefunc
            logger.info("Resetting Policy")
            pol_params = tf.get_default_session().run([nn.utils.parameters_to_vector(policy.parameters())])
            tf.get_default_session().run(tf.variables_initializer(policy.parameters()))
            pol_params_after = tf.get_default_session().run([nn.utils.parameters_to_vector(policy.parameters())])
            print ("pol_params:", np.linalg.norm(pol_params), "pol_params_after_reset:", np.linalg.norm(pol_params_after))
            logger.info("Resetting Valuefunc")
            tf.get_default_session().run(tf.variables_initializer(vfn.parameters()))

            tf.get_default_session().run(tf.variables_initializer(warmup_policy.parameters()))
            tf.get_default_session().run(tf.variables_initializer(warmup_vfn.parameters()))
            for p in warmup_policy.parameters(): p.invalidate()
            for p in warmup_vfn.parameters(): p.invalidate()
            for p in policy.parameters(): p.invalidate()
            for p in vfn.parameters(): p.invalidate()

        last_end = None
        drops = []

        evaluate(settings, 'pre-warm-up')
        returns_pre_warmup = testeval(policy, runners['collect'])
        if test:
            test_returns.append(returns_pre_warmup)
            test_summary['random'].append(returns_pre_warmup)
        t1 = time.time()
        trpo_time = 0

        logger.info('----------------------------- Warmup for %d iterations ------------------------' % warmup_n_iters)
        if decay == 'joint':
            logger.info('Joint train from a joint dataset')
        elif decay == 'taskid':
            Z = np.sum([float(i+1) for i in range(0, TASK_NUM)])
            prop = [float(taskid+1) / Z for taskid in range(TASK_NUM)]
            logger.info(f'Sampling prop={prop}, Z={Z}')
        elif decay == 'none':
            Z = TASK_NUM
            prop = [1. / TASK_NUM for _ in range(TASK_NUM)]
            logger.info(f'Sampling prop={prop}, Z={Z}')
        for i in range(warmup_n_iters):
            #exit(0)
            if TASK_NUM == 0 and not test and not model_load:
                logger.info('Break because TASK_NUM=0')
                break

            losses = deque(maxlen=warmup_n_model_iters)
            grad_norm_meter = AverageMeter()
            n_model_iters = warmup_n_model_iters
            drop_plot = 0
            if test and verbose:
                logger.info(f'warmup iter #{i}/{warmup_n_iters}, Do Not train Model during warmup of test time')
            if 'warmup_task_val_loss' not in taskres.keys():
                taskres['warmup_task_val_loss'] = [[] for _ in range(TASK_NUM)]

            if verbose: logger.info('Train Model for %d iterations' % n_model_iters)
            model_time = time.time()
            if not test or (test and have_data):
                for _ in range(n_model_iters):
                    if decay == 'joint':
                        samples = train_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                    else:
                        all_samples = []
                        for taskid in range(TASK_NUM):
                            samples_i = task_train_sets[taskid].sample_multi_step(int(FLAGS.model.train_batch_size*prop[taskid])+1, 1, FLAGS.model.multi_step)
                            all_samples.append(samples_i)
                        samples = np.concatenate(all_samples, axis=1).view(np.recarray)
                    _, train_loss, grad_norm = loss_mod.get_loss(
                        samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout,
                        fetch='train loss grad_norm')
                    losses.append(train_loss.mean())
                    grad_norm_meter.update(grad_norm)
                    # ideally, we should define an Optimizer class, which takes parameters as inputs.
                    # The `update` method of `Optimizer` will invalidate all parameters during updates.
                    for param in model.parameters():
                        param.invalidate()
                model_time = time.time() - model_time

                if i % FLAGS.model.validation_freq == 0:
                    task_val_loss = []
                    val_time = time.time()
                    for task_idx in range(TASK_NUM):
                        total_loss = []
                        for scan in range(FLAGS.rollout.n_dev_samples // FLAGS.model.dev_batch_size + 1):
                            samples = task_dev_sets[task_idx].sample_multi_step(FLAGS.model.dev_batch_size, 1, FLAGS.model.multi_step)
                            loss_i = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
                            total_loss.append(loss_i.mean())
                        total_loss = np.mean(total_loss)
                        task_val_loss.append(total_loss)
                        taskres['warmup_task_val_loss'][task_idx].append(total_loss)
                    val_time = time.time() - val_time
                    val_loss = np.mean(task_val_loss)
                    val_losses_warmup.append(val_loss)
                    train_losses_warmup.append(np.mean(losses))
                    if np.isnan(val_loss) or np.isnan(np.mean(losses)):
                        logger.info('nan! %s %s', np.isnan(val_loss), np.isnan(np.mean(losses)))
                    logger.info('# Warmup Iter %3d: Loss = [train = %.3f, dev = %.3f], after %d steps, grad_norm = %.6f, drop = %.2f, model_time=%d, trpo_time=%d, val_time=%d',
                                i, np.mean(losses), val_loss, n_model_iters, grad_norm_meter.get(), drop_plot, model_time, trpo_time, val_time)
                    logger.info(f'# task_val_loss: {task_val_loss}')

            if verbose: logger.info('Train policy for %d iterations' % warmup_n_policy_iters)
            trpo_time = time.time()
            for n_updates in range(warmup_n_policy_iters):
                if FLAGS.algorithm != 'MF' and FLAGS.warmup.start == 'buffer':
                    runners['train'].set_state(train_set.sample(FLAGS.plan.n_envs).state)
                else:
                    runners['train'].reset()

                data, ep_infos = runners['train'].run(policy, FLAGS.plan.n_trpo_samples)
                advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, x_velocity, begin_mark = runners['train'].compute_advantage(vfn, data,task)
                dist_mean, dist_std, vf_loss, plotinfo = algo.train(warmupent, data, advantages, values)
                trpo_warmup.append(plotinfo)
                returns = [info['return'] for info in ep_infos]
                if n_updates == 0:
                    if last_end is not None:
                        drop_plot = last_end - np.mean(returns)
                        drops.append(last_end - np.mean(returns))
                last_end = np.mean(returns)
                if n_updates == warmup_n_policy_iters-1:
                    logger.info('[TRPO] # %d: n_episodes = %d, returns: {mean = %.0f, std = %.0f}, '
                                'dist std = %.10f, dist mean = %.10f, vf_loss = %.3f',
                                n_updates, len(returns), np.mean(returns), np.std(returns) / np.sqrt(len(returns)),
                                dist_std, dist_mean, vf_loss)
            trpo_time = time.time() - trpo_time

            if i % FLAGS.warmup.n_evaluate_iters == 0 or i == warmup_n_iters-1:# and i != 0:
                real_eval, virt_eval = evaluate(settings, 'iteration')
                if 'warmup_real_eval' not in taskres.keys(): taskres['warmup_real_eval'] = []
                if 'warmup_virt_eval' not in taskres.keys(): taskres['warmup_virt_eval'] = []
                taskres['warmup_real_eval'].append(real_eval)
                taskres['warmup_virt_eval'].append(virt_eval)
                if test:
                    test_summary['warmupprocess'][TEST_TASK_NUM].append(real_eval)

        if not test:
            res = render(Monitor(make_env(FLAGS.env.id, task_config=task), f"./{setting}/{taskname}-task{TASK_NUM}-warmup/", force=True, video_callable=lambda episode_id: True), policy)
        else:
            res = render(Monitor(make_env(FLAGS.env.id, task_config=task), f"./{setting}/{taskname}-testtask{TEST_TASK_NUM}-warm{warmup_n_iters}-warmup/", force=True, video_callable=lambda episode_id: True), policy)
        taskres['warmup_monitor'] = [res]

        t2 = time.time()
        warmup_time = t2 - t1
        evaluate(settings, 'post-warm-up')
        returns_post_warmup = testeval(policy, runners['collect'])
        if test:
            test_returns.append(returns_post_warmup)
            test_summary['warmup'].append(returns_post_warmup)
            print ("warmupprocess:", test_summary['warmupprocess'][TEST_TASK_NUM])

        logger.info('Sync warmup policy and vfn and model')
        tf.get_default_session().run([sync_warmup_policy, sync_warmup_vfn, sync_warmup_model])
        for p in warmup_policy.parameters(): p.invalidate()
        for p in warmup_vfn.parameters(): p.invalidate()
        for p in warmup_model.parameters(): p.invalidate()
        for p in policy.parameters(): p.invalidate()
        task.parameters().invalidate()

        pol_params, warm_params = tf.get_default_session().run([nn.utils.parameters_to_vector(policy.parameters()), nn.utils.parameters_to_vector(warmup_policy.parameters())])
        print ("After WARMUP, pol_params_norm:", np.linalg.norm(pol_params), "warm_params_norm:", np.linalg.norm(warm_params))
        mod, warm_mod = tf.get_default_session().run([nn.utils.parameters_to_vector(model.parameters()), nn.utils.parameters_to_vector(warmup_model.parameters())])
        print ("mod_norm:", np.linalg.norm(mod), "warm_mod_norm:", np.linalg.norm(warm_mod))

        eval_rollout(runners['train'], warmup_policy, 'Use warmup policy to collect data from virtual env')
        warmup_collect_virt = []

        eval_rollout(runners['train'], policy, 'Use policy to collect data from virtual env')
        warmup_collect_real = []
        
        logger.info('--------------------------------------------- SLBO for %d outer stages -----------------------------------------' % slbo_n_stages)
        for T in range(slbo_n_stages):
            logger.info('-------- Starting Stage %d ---------', T)
            evaluate(settings, 'episode')

            # collect data
            if not test:
                logger.info('-------- Collect data from REAL env for %d samples --------' % FLAGS.rollout.n_train_samples)
                recent_train_set, ep_infos = runners['collect'].run(noise.make(policy), FLAGS.rollout.n_train_samples)
                recent_dev_set, _ = runners['dev'].run(noise.make(policy), FLAGS.rollout.n_dev_samples)
            else:
                logger.info('-------- Collect data from REAL env for %d samples --------' % 2000)
                recent_train_set, ep_infos = runners['collect2000'].run(noise.make(policy), 2000)
                recent_dev_set, _ = runners['dev'].run(noise.make(policy), FLAGS.rollout.n_dev_samples)

            logger.info('save setting dataset! trainset and devset!')
            if not test:
                pickle.dump(recent_train_set, open(f'./{setting}/{taskname}.trainset.task{TASK_NUM}.slbo{T}.pkl', 'wb'))
                pickle.dump(recent_dev_set, open(f'./{setting}/{taskname}.devset.task{TASK_NUM}.slbo{T}.pkl', 'wb'))

            # Add real data to task_train_sets and task_dev_sets
            #if not test:
            #    add_multi_step(recent_train_set, train_set)
            add_multi_step(recent_train_set, task_train_sets[TASK_NUM])
            add_multi_step(recent_dev_set, task_dev_sets[TASK_NUM])

            #if not test:
            #    states = recent_train_set.state
            #    mean = np.mean(states, axis=0)
            #    std = np.std(states, axis=0)
            #    min_ = np.min(states, axis=0)
            #    max_ = np.max(states, axis=0)
            #    states_stat = {"mean": mean, "std": std, "min": min_, "max": max_}

            # evaluate the surprisal of collected real data for model
            new_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
            add_multi_step(recent_train_set, new_set)
            losses_new = []
            for i in range(FLAGS.rollout.n_train_samples // FLAGS.model.dev_batch_size + 1):
                samples = new_set.sample_multi_step(FLAGS.model.dev_batch_size, 1, FLAGS.model.multi_step)
                loss = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
                loss = loss.mean()
                losses_new.append(loss)
            losses_new_mean = np.mean(losses_new)
            surprisal.append(losses_new_mean)
            logger.info(f'(surprisal) model loss on new collected data is {losses_new_mean}')

            add_multi_step(recent_train_set, train_set)
            add_multi_step(
                runners['dev'].run(noise.make(policy), FLAGS.rollout.n_dev_samples)[0],
                dev_set,
            )

            returns = np.array([ep_info['return'] for ep_info in ep_infos])
            if len(returns) > 0:
                logger.info("episode: %s", np.mean(returns))

            if T == 0:  # check
                samples = train_set.sample_multi_step(100, 1, FLAGS.model.multi_step)
                for i in range(FLAGS.model.multi_step - 1):
                    masks = 1 - (samples.done[i] | samples.timeout[i])[..., np.newaxis]
                    assert np.allclose(samples.state[i + 1] * masks, samples.next_state[i] * masks)

            normalizers.state.update(recent_train_set.state)
            normalizers.action.update(recent_train_set.action)
            normalizers.diff.update(recent_train_set.next_state - recent_train_set.state)

            if TASK_NUM == 0:  #In the 1st task, no warmup, but we validate loss of the random model
                samples = dev_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                loss = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
                loss = loss.mean()
                val_losses_warmup.append(loss)

            logger.info('SLBO for %d inner stages' % slbo_n_iters)
            model_time, trpo_time = 0, 0
            if 'slbo_task_val_loss' not in taskres.keys():
                taskres['slbo_task_val_loss'] = [[] for _ in range(TASK_NUM+1)]
            if decay == 'joint':
                logger.info('Joint train from a joint dataset')
            elif decay == 'taskid':
                Z = np.sum([float(i+1) for i in range(0, TASK_NUM+1)])
                prop = [float(taskid+1) / Z for taskid in range(TASK_NUM+1)]
                logger.info(f'Sampling prop={prop}, Z={Z}')
            elif decay == 'none':
                Z = TASK_NUM+1
                prop = [1. / float(Z) for _ in range(Z)]
                logger.info(f'Sampling prop={prop}, Z={Z}')
            for i in range(slbo_n_iters):
                if i % FLAGS.slbo.n_evaluate_iters == 0 or i == slbo_n_iters-1:# and i != 0:
                    # cur_actions = policy.eval('actions_mean actions_std', states=recent_states)
                    # kl_old_new = gaussian_kl(*ref_actions, *cur_actions).sum(axis=1).mean()
                    # logger.info('KL(old || cur) = %.6f', kl_old_new)
                    real_eval, virt_eval = evaluate(settings, 'iteration')
                    if 'slbo_real_eval' not in taskres.keys(): taskres['slbo_real_eval'] = []
                    if 'slbo_virt_eval' not in taskres.keys(): taskres['slbo_virt_eval'] = []
                    taskres['slbo_real_eval'].append(real_eval)
                    taskres['slbo_virt_eval'].append(virt_eval)

                losses = deque(maxlen=slbo_n_model_iters)
                grad_norm_meter = AverageMeter()
                n_model_iters = slbo_n_model_iters
                if verbose: logger.info('Train model %d iterations'% n_model_iters)
                model_time = time.time()
                for _ in range(n_model_iters):
                    if decay == 'joint':
                        samples = train_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                    else:
                        all_samples = []
                        sample_size = 0
                        for taskid in range(TASK_NUM+1):
                            samples_i = task_train_sets[taskid].sample_multi_step(int(FLAGS.model.train_batch_size*prop[taskid])+1, 1, FLAGS.model.multi_step)
                            all_samples.append(samples_i)
                            sample_size += int(FLAGS.model.train_batch_size*prop[taskid])+1
                        samples = np.concatenate(all_samples, axis=1).view(np.recarray)

                    _, train_loss, grad_norm = loss_mod.get_loss(
                        samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout,
                        fetch='train loss grad_norm')
                    losses.append(train_loss.mean())
                    grad_norm_meter.update(grad_norm)
                    # ideally, we should define an Optimizer class, which takes parameters as inputs.
                    # The `update` method of `Optimizer` will invalidate all parameters during updates.
                    for param in model.parameters():
                        param.invalidate()
                model_time = time.time() - model_time

                if i % FLAGS.model.validation_freq == 0:
                    task_val_loss = []
                    val_time = time.time()
                    for task_idx in range(TASK_NUM+1):
                        total_loss = []
                        for scan in range(FLAGS.rollout.n_dev_samples // FLAGS.model.dev_batch_size + 1):
                            samples = task_dev_sets[task_idx].sample_multi_step(FLAGS.model.dev_batch_size, 1, FLAGS.model.multi_step)
                            loss_i = loss_mod.get_loss(samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
                            total_loss.append(loss_i.mean())
                        total_loss = np.mean(total_loss)
                        task_val_loss.append(total_loss)
                        taskres['slbo_task_val_loss'][task_idx].append(total_loss)
                    val_loss = np.mean(task_val_loss)
                    val_time = time.time() - val_time
                    if np.isnan(val_loss) or np.isnan(np.mean(losses)):
                        logger.info('nan! %s %s', np.isnan(val_loss), np.isnan(np.mean(losses)))
                    logger.info('# SLBO Inner Iter %3d: Loss = [train = %.3f, dev = %.3f], after %d steps, grad_norm = %.6f, model_time=%d, trpo_time=%d, val_time=%d',
                                i, np.mean(losses), val_loss, n_model_iters, grad_norm_meter.get(), model_time, trpo_time, val_time)
                    logger.info(f'# task_val_loss: {task_val_loss}')
                    model_time, trpo_time = 0, 0
                    val_losses_slbo.append(val_loss)
                    train_losses_slbo.append(np.mean(losses))

                if verbose: logger.info('Train policy %d iterations'% slbo_n_policy_iters)
                trpo_time = time.time()
                for n_updates in range(slbo_n_policy_iters):
                    if FLAGS.algorithm != 'MF' and FLAGS.slbo.start == 'buffer':
                        runners['train'].set_state(train_set.sample(FLAGS.plan.n_envs).state)
                    else:
                        runners['train'].reset()

                    data, ep_infos = runners['train'].run(policy, FLAGS.plan.n_trpo_samples)
                    advantages, advantages_params, values, td, coef_mat, coef_mat_returns, reward_ctrl, x_velocity, begin_mark = runners['train'].compute_advantage(vfn, data, task)
                    dist_mean, dist_std, vf_loss, plotinfo = algo.train(max_ent_coef, data, advantages, values)
                    trpo_slbo.append(plotinfo)
                    returns = [info['return'] for info in ep_infos]
                    if n_updates == slbo_n_policy_iters-1:
                        logger.info('[TRPO] # %d: n_episodes = %d, returns: {mean = %.0f, std = %.0f}, '
                                    'dist std = %.10f, dist mean = %.10f, vf_loss = %.3f',
                                    n_updates, len(returns), np.mean(returns), np.std(returns) / np.sqrt(len(returns)),
                                    dist_std, dist_mean, vf_loss)
                trpo_time = time.time() - trpo_time
            
            if not test and (TASK_NUM) % FLAGS.ckpt.n_save_stages == 0:
                np.save(f'{FLAGS.log_dir}/{taskname}-stage-{TASK_NUM}', saver.state_dict())
                np.save(f'{FLAGS.log_dir}/{taskname}-final', saver.state_dict())
                res = render(Monitor(make_env(FLAGS.env.id, task_config=task), f"./{setting}/{taskname}-task{TASK_NUM}-slbo{T}/", force=True, video_callable=lambda episode_id: True), policy)
                if 'slbo_monitor' not in taskres.keys():
                    taskres['slbo_monitor'] = []
                taskres['slbo_monitor'].append(res)
            if not test and FLAGS.ckpt.n_save_stages == 1:
                pickle.dump(recent_train_set, open(f'{FLAGS.log_dir}/stage-{TASK_NUM}.inc-buf.pkl', 'wb'))
            if test:
                returns_post_slbo_update = testeval(policy, runners['collect'])
                test_returns.append(returns_post_slbo_update)
                real_eval, virt_eval = evaluate(settings, 'iteration')
                test_summary['slbo'][TEST_TASK_NUM].append(real_eval)
                test_summary[f'slbo{T+1}'].append(returns_post_slbo_update)
                res = render(Monitor(make_env(FLAGS.env.id, task_config=task), f"./{setting}/{taskname}-testtask{TEST_TASK_NUM}-slbo{T}/", force=True, video_callable=lambda episode_id: True), policy)
                print ('test_summary_slbo:', test_summary['slbo'][TEST_TASK_NUM])

            if not test:
                np.save(f'{setting}/{taskname}.task{TASK_NUM}.saver', saver.state_dict())
                np.save(f'{setting}/{taskname}.final.saver', saver.state_dict())

        if init_generator and TASK_NUM==0:
            print ('finished init generator!')
            exit(0)

        pol_params, warm_params = tf.get_default_session().run([nn.utils.parameters_to_vector(policy.parameters()), nn.utils.parameters_to_vector(warmup_policy.parameters())])
        print ("After SLBO, pol_params_norm:", np.linalg.norm(pol_params), "warm_params_norm:", np.linalg.norm(warm_params))

        eval_rollout(runners['train'], policy, 'Use optimal policy to collect data from real env')
        optimal_collect_real = []

        t3 = time.time()
        slbo_time = t3 - t2
        evaluate(settings, 'post-slbo')
        logger.info(f'Warmup time = {warmup_time}, SLBO time = {slbo_time}')

        alltaskres.append(taskres)
        if not test:
            pickle.dump(alltaskres, open(f'{setting}/{taskname}-alltaskres.info.pkl', 'wb'))
            pickle.dump(all_task_parameter, open(f'{setting}/all_task_parameter.pkl', 'wb'))
        else:
            pickle.dump(alltaskres, open(f'{setting}/{taskname}-alltaskres.info.pkl.{testparam}', 'wb'))
            pickle.dump(all_task_parameter, open(f'{setting}/all_task_parameter.pkl.{testparam}', 'wb'))

        eval_rollout(runners['train'], warmup_policy, 'Use warmup policy to collect data from virtual env')
        if not test:
            #if TASK_NUM > 0:
            if TASK_NUM > -1:
                task_params_before, final_grad, advtask_info = advtask.train(runners['train_copy'], runners['collect_copy'], warmup_collect_virt, warmup_collect_real, optimal_collect_real, returns_pre_warmup, val_losses_warmup, val_losses_slbo, train_losses_warmup, train_losses_slbo, surprisal, trpo_warmup, trpo_slbo, fout, infofilename, extra_runners)

        # first task or maxstep, update the model
        if not test and (TASK_NUM == 0 or TASK_NUM % maxstep == 0):
            logger.info(f"task_num={TASK_NUM}, sync_model_to_lazymodel")
            tf.get_default_session().run(sync_model_to_lazymodel)

        if test:
            pickle.dump(test_summary, open(f'{setting}/test_summary.pkl.{testparam}', 'wb'))
            TEST_TASK_NUM += 1
            TASK_NUM = train_tasknum
            #task_train_sets[TASK_NUM].clear()
            #task_dev_sets[TASK_NUM].clear()
            for tt in range(TASK_NUM+1):
                task_train_sets[tt].clear()
                task_dev_sets[tt].clear()
            train_set.clear()
            load_data_during_test()
            continue

        task_params_after = task_params_before + final_grad * alpha
        task.set_parameters(task_params_after)

        if not test:
            advtask_info['alpha'].append(alpha)
            with open(infofilename, 'wb') as handle:
                pickle.dump(advtask_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print ('>>>>>>dump')

        TASK_NUM += 1
        time_end = time.time()
        print (f"Task Done! Total Time Consumed for 1 task = {time_end - time_start}s")



if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
