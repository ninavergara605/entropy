import numpy as np


def calculate_ent_disp(trans_matrix):
        prob_matrix, trans_count = calculate_probability_matrix(trans_matrix)    
        entropy_matrix = calculate_matrix_entropy(prob_matrix)
        total_entropy = calculate_entropy_total(entropy_matrix, prob_matrix)
        return total_entropy, trans_count

def calculate_probability_matrix(trans_matrix):
    # Divides each transition count by the total number of cells
    # Returns a matrix of probabilities
    trans_count = np.sum(trans_matrix, dtype='float64')
    prob_matrix = trans_matrix/trans_count
    return prob_matrix, trans_count

def calculate_matrix_entropy(m):
    #partial application of shannon's entropy formula to be used on a probability matrix
    entropies =np.multiply(m, np.log2(1/m), where=[m !=0], dtype='float64')
    entropies[np.logical_or(entropies == np.inf, entropies == -np.inf)] = 0
    return entropies 

def calculate_entropy_total(trans_entropy, trans_prob):
    raw_entropy_total = trans_entropy.sum()
    row_entropy, col_entropy = get_row_col_entropy(trans_prob)
    
    # Subtract row and col entropy from raw total
    entropy_numerator = col_entropy + row_entropy - raw_entropy_total
    # Calculate the mean of row and col entropies
    entropy_denominator = (col_entropy + row_entropy) / 2
    total_entropy = entropy_numerator / entropy_denominator    
    return total_entropy

def get_row_col_entropy(trans_prob):
    # calculates the entropy of column and row sums individually
    # returns an array containing the sums of the column and row entropy matrices
    col_totals = trans_prob.sum(axis=0, dtype='float64')
    row_totals = trans_prob.sum(axis=1, dtype='float64')       
    row_col_ent = [np.sum(
                        calculate_matrix_entropy(m)
                        ,dtype='float64') 
                        for m in [row_totals, col_totals]
                        ]
    return row_col_ent







    
