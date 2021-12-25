import numpy as np
from numpy.linalg.linalg import norm


def get_trans_matrix(trans_count):
    row_sums = np.sum(trans_count, axis=1)
    trans_prob = trans_count / row_sums[:, None]
    return trans_prob


def calc_stationary_distribution(trans_prob):
    S, U = np.linalg.eig(trans_prob.T)
    stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary = stationary / np.sum(stationary)
    return stationary.real


def calc_entropy(trans_prob, stationary_distr=np.array([])):
    s = len(trans_prob)
    if len(stationary_distr) == 0:
        stationary_distr = np.repeat(1 / s, s)

    entropy_matrix = -stationary_distr[:, None] * (trans_prob * np.log2(trans_prob))
    entropy_matrix = np.nan_to_num(entropy_matrix)
    total_entropy = np.sum(entropy_matrix)

    return total_entropy


def normalize(total_entropy, s, exclude_diag=False):
    max_trans_prob = np.reshape(np.repeat(1 / s, s * s), (s, s))
    if exclude_diag:
        np.fill_diagonal(max_trans_prob, 0)
    max_entropy = calc_entropy(max_trans_prob)
    norm_entropy = total_entropy / max_entropy
    return norm_entropy


trans_prob = np.array([[0.5, 0.5, 0]
                          , [0.25, 0.5, 0.25]
                          , [0, 0.5, 0.5]])

trans_count = np.array([[0, 1, 0, 0]
                           , [0, 0, 1, 0]
                           , [0, 1, 1, 1]
                           , [0, 1, 1, 1]])

trans_prob = get_trans_matrix(trans_count)
_stationary_distr = calc_stationary_distribution(trans_prob)
np.fill_diagonal(trans_prob, 0)

entropy = calc_entropy(trans_prob, stationary_distr=_stationary_distr)
norm_ent = normalize(entropy, len(trans_prob), exclude_diag=True)
