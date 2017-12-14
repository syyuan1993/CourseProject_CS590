import tensorflow as tf
import numpy as np
import gym


class Environment(object):
    def __init__(self, run_dir, env_name):
        self.name = env_name
        self.gym = gym.make(self.name)


        self.random_initialization = True
        self._connect()
        self._train_params()
        self.run_dir = run_dir

    def _step(self, action):
        action = np.squeeze(action)
        self.t += 1
        result = self.gym.step(action)
        self.state, self.reward, self.done, self.info = result[:4]
        if self.random_initialization:
            self.qpos, self.qvel = self.gym.env.model.data.qpos.flatten(), self.gym.env.model.data.qvel.flatten()
            return np.float32(self.state), np.float32(self.reward), self.done, np.float32(self.qpos), np.float32(self.qvel)
        else:
            return np.float32(self.state), np.float32(self.reward), self.done

    def step(self, action, mode):
        qvel, qpos = [], []
        if mode == 'tensorflow':
            if self.random_initialization:
                state, reward, done, qval, qpos = tf.py_func(self._step, inp=[action], Tout=[tf.float32, tf.float32, tf.bool, tf.float32, tf.float32], name='env_step_func')
            else:
                state, reward, done = tf.py_func(self._step, inp=[action],
                                                 Tout=[tf.float32, tf.float32, tf.bool],
                                                 name='env_step_func')

            state = tf.reshape(state, shape=(self.state_size,))
            done.set_shape(())
        else:
            if self.random_initialization:
                state, reward, done, qvel, qpos = self._step(action)
            else:
                state, reward, done = self._step(action)

        return state, reward, done, 0., qvel, qpos

    def reset(self, qpos=None, qvel=None):
        self.t = 0
        self.state = self.gym.reset()
        if self.random_initialization and qpos is not None and qvel is not None:
            self.gym.env.set_state(qpos, qvel)
        return self.state

    def get_status(self):
        return self.done

    def get_state(self):
        return self.state

    def render(self):
        self.gym.render()

    def _connect(self):
        self.state_size = self.gym.observation_space.shape[0]
        self.action_size = self.gym.action_space.shape[0]
        self.action_space = np.asarray([None]*self.action_size)
        self.qpos_size = self.gym.env.data.qpos.shape[0]
        self.qvel_size = self.gym.env.data.qvel.shape[0]
        print('state size is: %d, action size is: %d, qpos_size is: %d, qvel_size is: %d\n'% (self.state_size, self.action_size, self.qpos_size, self.qvel_size))

    def _train_params(self):
        self.trained_model = None
        self.train_mode = True
        self.n_train_iters = 1000000
        self.n_episodes_test = 10
        self.test_interval = 1000
        self.n_steps_test = 1000
        self.vis_flag = False
        self.save_models = True
        self.config_dir = None
        self.continuous_actions = True

        # Main parameters to play with:
        self.expert_data = 'expert_trajectories/hopper_er.bin'  # need to be changed
        self.er_agent_size = 5000 # default to be 50k
        self.prep_time = 1000 # default to be 1k
        self.collect_experience_interval = 15 # default to be 15
        self.n_steps_train = 10 # default to be 10
        self.discr_policy_itrvl = 100 # default to be 100
        self.gamma = 0.99 # default to be 0.99
        self.batch_size = 70 # default to be 70
        self.weight_decay = 1e-7 # default to be 1e-7
        self.policy_al_w = 1e-2 # default to be 1e-2
        self.policy_tr_w = 1e-4 # default 1e-4
        self.policy_accum_steps = 7 # default 7
        self.total_trans_err_allowed = 1000 # default 1k
        self.temp = 1. # default 1.
        self.cost_sensitive_weight = 1.0 # sensitive of discriminator loss, weighting between true data and generated data, default to be 0.8
        self.noise_intensity = 6. # default 6.
        self.do_keep_prob = 0.75 # default 0.75

        # Hidden layers size
        self.fm_size = 100 # default 100
        self.d_size = [200, 100] # default [200, 100]
        self.p_size = [100, 50] # default [100,50]

        # Learning rates
        self.fm_lr = 1e-4 # forward model 1e-4
        self.d_lr = 1e-4 # discriminator # default to be 1e-3
        self.p_lr = 1e-4 # policy 1e-4

        # type of GAN:
        self.gan_type = 3
        # 1 means vanilla
        # 2 means WGAN
        # 3 means WGAN-GP
        self.LAMBDA = 10 # parameter for WGAN-GP
        # self.results_fname = 'Acrobot'
        self.results_fname = 'initializer'

        self.GRU = True


