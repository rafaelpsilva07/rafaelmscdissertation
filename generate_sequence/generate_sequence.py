import numpy as np

from itertools import product
from random import choices

from composipy import OrthotropicMaterial, LaminateProperty


def _layup_permutations(n_plies):
    '''
    given number of plies, returns the permutation of stacking sequences    
    '''
    angle = [0, 1, 2, 3] # --> 0, 90, 45, -45
    angles = (angle for i in range(n_plies))

    sequences = product(*angles)
    sequences = list(sequences)
    return sequences


def _random_layups(n_plies):
    '''
    given number of plies, and the number    
    '''


def _is_balanced(stacking):
    n2 = stacking.count(2)
    n3 = stacking.count(3)
    if n2 == n3:
        return True
    else:
        return False


def _contiguity(stacking):
    '''https://elib.dlr.de/107894/1/__Bsfait00_fa_Archive_IB_2016_IB_2016_173_MA%20Werthen.pdf
    
    According to NIU no more than 4 plies
     '''

    n = len(stacking)

    for i in range(n-4):
        i0 = stacking[i]
        i1 = stacking[i+1]
        i2 = stacking[i+2]
        i3 = stacking[i+3]
        i4 = stacking[i+4]
        if i0 == i1 == i2 == i3:
            return False

    # Contiguity of middle plies
    if stacking[-1] == stacking[-2] == stacking[-3]:
        False
    return True


def _10p_rule(stacking):
    n0 = stacking.count(0)
    n1 = stacking.count(1)
    n2 = stacking.count(2)
    n3 = stacking.count(3)
    n = len(stacking)
    n_10p = n * 0.1

    if (n0 < n_10p
            or n1 < n_10p
            or n2 < n_10p
            or n3 < n_10p):
        return False
    else:
        return True


def generate_sequence(n_plies):
    '''
    Given number of plies and filtering rules, returns a new permutation.
    '''
    permutations = _layup_permutations(n_plies)
    accepted_sequences = []
    for sequence in permutations:
        if (_is_balanced(sequence)
                and _contiguity(sequence)
                and _10p_rule(sequence)
                ):
            accepted_sequences.append(sequence)
    return accepted_sequences


def calc_dist(p1, p2):
    return np.linalg.norm(p1-p2)


def generate_best_laminate(xi1D, xi3D, n_plies):
    '''
    Given number of plies and filtering rules, returns a new permutation.
    '''
    permutations = _layup_permutations(n_plies)
    dist = 9999.9
    p1 = np.array([xi1D, xi3D])
    mat = OrthotropicMaterial(1000, 1000, 0.3, 500, 0.1) #Dummy

    for sequence in permutations:
        if (_is_balanced(sequence)
                and _contiguity(sequence)
                and _10p_rule(sequence)
                ):
            

            stack = convert_sequence_to_symmetric_laminate(sequence)
            lam = LaminateProperty(stack, mat)
            p2 = np.array([lam.xiD[0], lam.xiD[2]])
            cur_dist = calc_dist(p1, p2)
            if cur_dist < dist:
                dist = cur_dist
                best_laminate = stack 
    return best_laminate


def generate_random_sequence(n_plies, n_sequences=500):
    '''
    Given number of plies and filtering rules, returns a new permutation.
    '''

    cur_combination_seq = {
    5: 42,
    6: 156,
    7: 470,
    8: 1342,
    9: 3626,
    10: 9548,
    11: 14204,
    12: 42162
    }

    if n_plies in cur_combination_seq.keys():
        if cur_combination_seq[n_plies] < n_sequences:
            return generate_sequence(n_plies)

    accepted_sequences = []
    while len(accepted_sequences) <n_sequences:
        combination = [choices([0, 1, 2, 3])[0] for i in range(n_plies)]
        if (_is_balanced(combination)
                and _contiguity(combination)
                and _10p_rule(combination)
                ):
            accepted_sequences.append(combination)
    return accepted_sequences


def convert_sequence_to_symmetric_laminate(sequence, isodd=False):
    convert = {0: 0,
               1: 90,
               2: 45,
               3: -45}
               
    converted_sequence = []
    for v in sequence:
        converted_sequence.append(convert[v])

    if isodd:
        converted_sequence += converted_sequence[::-1][1::] # invert and removes one middle ply
    else:
        converted_sequence += converted_sequence[::-1]
    return converted_sequence


def isodd(n):
    '''https://stackoverflow.com/questions/21837208/check-if-a-number-is-odd-or-even-in-python'''

    if n & 1:
        return True
    else:
        return False


def generate_laminate_sequences(n_plies, n_sequences=None):
    n = round((n_plies+0.1)/2)
    odd = isodd(n_plies)

    if n_sequences is None:
        sequences = generate_sequence(n)
    else:
        sequences = generate_random_sequence(n, n_sequences=n_sequences)
    new_sequences = []
    for s in sequences:
        new_sequences.append(
            convert_sequence_to_symmetric_laminate(s, isodd=odd)
        )

    return new_sequences


def get_best_laminate_random(xi1D, xi3D, n_plies, n_sequences=500):
    sequences = generate_laminate_sequences(n_plies, n_sequences)

    dist = 9999.9
    p1 = np.array([xi1D, xi3D])
    mat = OrthotropicMaterial(1000, 1000, 0.3, 2000, 0.1)

    for sequence in sequences:
        lam = LaminateProperty(sequence, mat)
        p2 = np.array([lam.xiD[0], lam.xiD[2]])
        cur_dist = calc_dist(p1, p2)
        if cur_dist < dist:
            dist = cur_dist
            best_laminate = sequence
    return best_laminate
