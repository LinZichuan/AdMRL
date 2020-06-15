import time
import os
import yaml
from subprocess import check_output, CalledProcessError
from lunzi.config import BaseFLAGS, expand, parse
from lunzi.Logger import logger, FileSink


class FLAGS(BaseFLAGS):
    _initialized = False

    use_prev = True
    seed = 100
    log_dir: str = None
    run_id: str = None
    algorithm = 'OLBO'  # possible options: OLBO, baseline, MF

    class task(BaseFLAGS):
        n_iters = 100 #5
        n_iters_test = 40
        method = 'random' # possibly 'random' or 'fixed'
        fixed_velocities = []
        reset_policy = True
        skip_policy = 'none' # possibly 'none' or 'drop-mean' or 'shadow-reward-variance' or 'shadow-state-error'
        skip_window = 4
        reward = 'none' # abs or quadratic or linear
        scaler = 'clip' # clip or none or tanh or sigmoid
        goalname = 'normal' #normal or reverse

    class adv(BaseFLAGS):
        use = True
        lr = 1
        n_samples = 8000

    class warmup(BaseFLAGS):
        n_iters = 20 #n_warmup=40
        n_policy_iters = 20  #n_policy=40
        n_model_iters = 1000 #n_model=100
        n_evaluate_iters = 5
        start = 'reset' # possibly 'buffer'
        n_shadow_models = 1
        n_shadow_model_iters = 100

    class slbo(BaseFLAGS):
        n_iters = 20 #n_iters=20 this number is from SLBO paper Appendix F.3
        n_policy_iters = 20  #n_policy=40
        n_model_iters = 1000 #n_model=100
        n_stages = 1 #100
        n_stages_test = 3 #100
        n_evaluate_iters = 5
        opt_model = False
        start = 'reset'  # possibly 'buffer'

    class plan(BaseFLAGS):
        max_steps = 200 #500
        n_envs = None
        n_trpo_samples = 8000

        @classmethod
        def finalize(cls):
            if cls.n_envs is None:
                cls.n_envs = cls.n_trpo_samples // cls.max_steps
            assert cls.n_envs * cls.max_steps == cls.n_trpo_samples

    class env(BaseFLAGS):
        #id = 'HalfCheetah-v2'
        id = 'HalfCheetahTask-v2'
        #id = 'Ant2DTask-v2'

    class rollout(BaseFLAGS):
        normalizer = 'policy'
        max_buf_size = 5000000
        n_train_samples = 8000 #10000
        n_dev_samples = 8000
        n_test_samples = 8000 #10000

        @classmethod
        def finalize(cls):
            cls.n_dev_samples = cls.n_dev_samples or cls.n_train_samples

    class ckpt(BaseFLAGS):
        n_save_stages = 1 #10
        model_load = None
        policy_load = None
        buf_load = None
        buf_load_index = 0
        base = '/tmp/admrl/logs'
        warm_up = None
        off_by_one = False

        @classmethod
        def finalize(cls):
            for key, value in cls.as_dict().items():
                if isinstance(value, str):
                    setattr(cls, key, expand(value))

    class OUNoise(BaseFLAGS):
        theta = 0.15
        sigma = 0.3

    class model(BaseFLAGS):
        #hidden_sizes = [500, 500]
        hidden_sizes = [1000, 1000]
        loss = 'L2'  # possibly L1, L2, MSE, G
        G_coef = 0.5
        multi_step = 1
        lr = 1e-3
        weight_decay = 1e-5
        validation_freq = 1
        optimizer = 'Adam'
        train_batch_size = 256
        dev_batch_size = 1024

    class policy(BaseFLAGS):
        hidden_sizes = [32, 32]
        init_std = 1.

    class PPO(BaseFLAGS):
        n_minibatches = 32
        n_opt_epochs = 10
        ent_coef = 0.005
        lr = 3e-4
        clip_range = 0.2

    class TRPO(BaseFLAGS):
        cg_damping = 0.1
        n_cg_iters = 10
        max_kl = 0.01
        vf_lr = 1e-3
        n_vf_iters = 5
        ent_coef = 0.005

    class runner(BaseFLAGS):
        lambda_ = 0.95
        gamma = 0.99
        max_steps = 200 #500

    @classmethod
    def set_seed(cls):
        if cls.seed == 0:  # auto seed
            cls.seed = int.from_bytes(os.urandom(3), 'little') + 1  # never use seed 0 for RNG, 0 is for `urandom`
        logger.warning("Setting random seed to %s", cls.seed)

        import numpy as np
        import tensorflow as tf
        import torch
        import random
        np.random.seed(cls.seed)
        tf.set_random_seed(np.random.randint(2**30))
        torch.manual_seed(np.random.randint(2**30))
        random.seed(np.random.randint(2**30))
        torch.cuda.manual_seed_all(np.random.randint(2**30))
        torch.backends.cudnn.deterministic = True

    @classmethod
    def finalize(cls):
        log_dir = cls.log_dir
        if log_dir is None:
            run_id = cls.run_id
            if run_id is None:
                run_id = time.strftime('%Y-%m-%d_%H-%M-%S')

            log_dir = os.path.join(cls.ckpt.base, run_id)
            cls.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # hack: don't copy code
        # for t in range(60):
        #     try:
        #         cls.commit = check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        #         check_output(['git', 'add', '.'])
        #         check_output(['git', 'checkout-index', '-a', '-f', f'--prefix={log_dir}/src/'])
        #         break
        #     except CalledProcessError:
        #         pass
        #     time.sleep(1)
        # else:
        #     raise RuntimeError('Failed after 60 trials.')

        yaml.dump(cls.as_dict(), open(os.path.join(log_dir, 'config.yml'), 'w'), default_flow_style=False)
        open(os.path.join(log_dir, 'diff.patch'), 'w').write(
            check_output(['git', '--no-pager', 'diff', 'HEAD']).decode('utf-8'))

        logger.add_sink(FileSink(os.path.join(log_dir, 'log.json')))
        logger.info("log_dir = %s", log_dir)

        cls.set_frozen()


parse(FLAGS)

