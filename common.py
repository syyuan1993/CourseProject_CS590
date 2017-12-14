import cPickle
import tensorflow as tf
import numpy as np
import h5py


def save_params(fname, saver, session):
    saver.save(session, fname)


def load_er(fname, batch_size, history_length, traj_length):
    f = file(fname, 'rb')
    er = cPickle.load(f)
    er.batch_size = batch_size
    er = set_er_stats(er, history_length, traj_length)
    return er

# def load_er_(fname, batch_size, history_length, traj_length):
#     f = file(fname, 'rb')
#     a = cPickle.load(f)
#     obs = np.array([])
#     for i in len(a):
#         obs = np.concatenate(obs, a[i]['ob'])
#
#     er.batch_size = batch_size
#     er = set_er_stats(er, history_length, traj_length)
#     return er


# def load_er(fname, batch_size, history_length, traj_length):
#     f = file('expert_trajectories/hopper_er.bin', 'rb')
#     er = cPickle.load(f)
#     with h5py.File(fname, 'r') as f:
#         # Read data as written by vis_mj.py
#         full_dset_size = f['obs_B_T_Do'].shape[0] # full dataset size
#         dset_size = min(full_dset_size, traj_length) if traj_length is not None else full_dset_size
#         er.batch_size = batch_size
#
#         exobs_B_T_Do = f['obs_B_T_Do'][:dset_size,...][...]
#         exa_B_T_Da = f['a_B_T_Da'][:dset_size,...][...]
#         exr_B_T = f['r_B_T'][:dset_size,...][...]
#         exlen_B = f['len_B'][:dset_size,...][...]
#         # Stack everything together
#     # start_times_B = np.random.RandomState(0).randint(0, data_subsamp_freq, size=exlen_B.shape[0])
#     #
#     # exobs_Bstacked_Do = np.concatenate(
#     #         [exobs_B_T_Do[i, start_times_B[i]:l:1, :] for i, l in enumerate(exlen_B)],
#     #         axis=0)
#     # exa_Bstacked_Da = np.concatenate(
#     #         [exa_B_T_Da[i, start_times_B[i]:l:1, :] for i, l in enumerate(exlen_B)],
#     #         axis=0)
#     # ext_Bstacked = np.concatenate(
#     #         [np.arange(start_times_B[i], l, step=1) for i, l in enumerate(exlen_B)]).astype(float)
#
#     er.states = exobs_B_T_Do
#     er.actions = exa_B_T_Da
#     er = set_er_stats(er, history_length, traj_length)
#     return er

def set_er_stats(er, history_length, traj_length):
    state_dim = er.states.shape[-1]
    action_dim = er.actions.shape[-1]
    er.prestates = np.empty((er.batch_size, history_length, state_dim), dtype=np.float32)
    er.poststates = np.empty((er.batch_size, history_length, state_dim), dtype=np.float32)
    er.traj_states = np.empty((er.batch_size, traj_length, state_dim), dtype=np.float32)
    er.traj_actions = np.empty((er.batch_size, traj_length-1, action_dim), dtype=np.float32)
    er.states_min = np.min(er.states[:er.count], axis=0)
    er.states_max = np.max(er.states[:er.count], axis=0)
    er.actions_min = np.min(er.actions[:er.count], axis=0)
    er.actions_max = np.max(er.actions[:er.count], axis=0)
    er.states_mean = np.mean(er.states[:er.count], axis=0)
    er.actions_mean = np.mean(er.actions[:er.count], axis=0)
    er.states_std = np.std(er.states[:er.count], axis=0)
    er.states_std[er.states_std == 0] = 1
    er.actions_std = np.std(er.actions[:er.count], axis=0)
    return er


def re_parametrization(state_e, state_a):
    nu = state_e - state_a
    nu = tf.stop_gradient(nu)
    return state_a + nu, nu


def normalize(x, mean, std):
    return (x - mean)/std


def denormalize(x, mean, std):
    return x * std + mean


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
