import numpy as np
from lukas_implementation.calculate_entropy_lukas import calculate_ent_disp

'''
Heather's version implemented:
    simulate random transition matricies and calculate entropies
    use mean to normalize subject data

    Added a steady state detection alogrithm to ensure that the mean entropy is staple
    Still returns values greater than 1 after normalization for some test cases
'''

class EntropySimulation:
    
    
    def __init__(self
                ,total_roi
                ):
        self._total_roi = total_roi
        self._choices = np.arange(total_roi)
        
    def simulation_dispatch(self, repeat, trans_count):
        trans_sequences = self.generate_sequences(repeat, trans_count)
        trans_matrices = EntropySimulation.get_trans_matrices(self._total_roi, trans_sequences)
        entropies = [calculate_ent_disp(trans_matrix)[0] for trans_matrix in trans_matrices]
        return entropies

    def generate_sequences(self, repeat, trans_count):  
        empty_sequences = [np.zeros((trans_count, 2)) for _ in range(repeat)]
        transition_seqs = []
        for empty_seq in empty_sequences:
            rand_trans = EntropySimulation.get_trans_seq(empty_seq
                                                        ,self._choices
                                                        ,trans_count)
            transition_seqs.append(rand_trans)
        return transition_seqs

    @staticmethod
    def get_trans_seq(empty_seq, choices, size):
        verified_sequence = False
        while verified_sequence == False:
            sequence = empty_seq
            for i in range(0, size):
                sequence[i] = EntropySimulation.get_transition(choices)
            
            missing_choice = np.setdiff1d(choices, sequence[:,0])
            if not any(missing_choice):
                verified_sequence = True
        return sequence

    @staticmethod
    def get_transition(choices):
        transition = [0,0]
        while transition[0] == transition[1]:
            transition = [np.random.choice(choices) for _ in range(2)]
        return transition
    
    @staticmethod
    def get_trans_matrices(total_roi, trans_seqs):
        num_seq = len(trans_seqs)
        trans_matrices = np.array([np.zeros((total_roi, total_roi)) for _ in range(num_seq)])

        for i in range(num_seq):
            seq = trans_seqs[i]
            trans_counts = zip(
                            *np.unique(seq
                            ,axis=0
                            ,return_counts=True
                            )) 
            
            for (k,j), count in trans_counts:
                trans_matrices[i][int(k),int(j)] = count
        return trans_matrices    
    
    
    
    '''
    def get_std(self, all_trans_matrices, choices):
        prob_matrices = np.array([calculate_probability_matrix(m) for m in all_trans_matrices])
        indexes = [(m,n) for m in self._choices for n in self._choices if m != n]
        
        _all = []
        for matrix in prob_matrices:
            trans = no_trans = 0
            for_std = []
            poss_choices = choices
            for i in choices:
                poss_choices = poss_choices[poss_choices!= i]
                for j in poss_choices:
                    cell, inverse_cell = matrix[i,j], matrix[j,i]
                    if  cell == 0 and inverse_cell == 0:
                        no_trans+= 1
                    else:
                        trans +=1
                    for_std.append(((trans+no_trans), cell+inverse_cell))
            _all.append(for_std)
        return _all
    '''