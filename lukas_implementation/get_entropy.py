import numpy as np
from lukas_implementation.calculate_entropy_lukas import calculate_ent_disp
from lukas_implementation.entropy_simulation import EntropySimulation
from collections import defaultdict


class GetEntropy:

    def __init__(self
                 , trans_matrices=None
                 , group=None
                 ):
        self._trans_matrices = trans_matrices
        self._total_roi = len(self._trans_matrices[0])
        self._entropy_totals = []
        self._trans_counts = []
        self._group = group
        self.result = self.get_entropy()

    def get_entropy(self):
        subject_entropies = self.get_subject_entropy()
        normalize_by = self.get_simulated_entropy()
        return subject_entropies, normalize_by

    def get_simulated_entropy(self):
        simulation = EntropySimulation(self._total_roi)
        trans_count_range = self.calculate_trans_count_range()
        sim_mean_std_final = defaultdict(dict)

        for trans_count in trans_count_range:
            entropies = simulation.simulation_dispatch(100, trans_count)
            mean_variance = GetEntropy.check_mean_variance(entropies)

            while mean_variance > 0.1:
                for i in range(1, 20):
                    new_entropies = simulation.simulation_dispatch(20, trans_count)
                    entropies.append(new_entropies)
                    mean_variance = GetEntropy.check_mean_variance(entropies)

            sim_mean_std_final[trans_count]['mean'] = np.mean(entropies)
            sim_mean_std_final[trans_count]['std'] = np.std(entropies)

    @staticmethod
    def check_mean_variance(entropies):
        means = np.zeros(50)
        for i in range(1, 50):
            means[i] = np.mean(entropies[:-i], dtype=np.float64)
        mean_std = np.std(means)
        return mean_std

    def get_subject_entropy(self):
        for trans_matrix in self._trans_matrices:
            entropy, trans_count = calculate_ent_disp(trans_matrix)
            self._entropy_totals.append(entropy)
            self._trans_counts.append(trans_count)

    def calculate_trans_count_range(self):
        _min = self._total_roi
        _max = max(self._trans_counts) + 1
        trans_count_range = np.arange(_min, _max)
        return trans_count_range

    def normalize_entropy(sim_info, entropy):
        normalized = (entropy - sim_info['mean']) / sim_info['std']
        return normalized


ex_a_raw = [[0, 1, 0, 1, 1, 0]
    , [0, 0, 1, 1, 1, 3]
    , [1, 0, 0, 1, 1, 0]
    , [0, 3, 1, 0, 1, 1]
    , [1, 1, 0, 0, 0, 0]
    , [1, 0, 0, 3, 0, 0]]

ex_b_raw = [[0, 0, 0, 1, 0, 3]
    , [2, 0, 0, 1, 4, 0]
    , [1, 0, 0, 2, 0, 0]
    , [0, 4, 0, 0, 0, 0]
    , [1, 2, 0, 0, 0, 0]
    , [0, 0, 3, 0, 0, 0]]

matrix_b = np.array([np.array(m) for m in ex_b_raw])
matrix_a = np.array([np.array(m) for m in ex_a_raw])
GetEntropy(trans_matrices=[matrix_a, matrix_b])
