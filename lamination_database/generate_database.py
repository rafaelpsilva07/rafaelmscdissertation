import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from random import choices, random

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


def _layup_permutations_product(n_plies):
    '''
    given number of plies, returns the permutation of stacking sequences    
    '''
    angle = [0, 1, 2, 3] # --> 0, 90, 45, -45
    angles = (angle for i in range(n_plies))

    sequences = product(*angles)
    #sequences = list(sequences)
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

    p0, p90, p45 = random(), random(), random()
    accepted_sequences = []
    while len(accepted_sequences) <n_sequences:
        combination = [choices([0, 1, 2, 3], weights=(p0, p90, p45, p45)
                               )[0] for i in range(n_plies)]
        if (_is_balanced(combination)
                and _contiguity(combination)
                and _10p_rule(combination)
                ):
            accepted_sequences.append(combination)
        p0, p90, p45 = random(), random(), random()
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


def get_best_laminate_random(xi1D, xi3D, n_plies, n_sequences=500, detailed=False):
    sequences = generate_laminate_sequences(n_plies, n_sequences)

    dist = 9999.9
    p1 = np.array([xi1D, xi3D])
    mat = OrthotropicMaterial(1000, 1000, 0.3, 2000, 0.1)

    rankedsolutions = []
    for s in sequences:
        lam = LaminateProperty(s, mat)
        p2 = np.array([lam.xiD[0], lam.xiD[2]])
        rankedsolutions.append(
            (calc_dist(p1, p2), s)
        )
    rankedsolutions.sort()

    if not detailed:
        return rankedsolutions[0][1]
    else:
        print('BEST SOLUTION IS')
        print(rankedsolutions[0])

        distances_sorted = []
        for d, s in rankedsolutions:
            distances_sorted.append(d)
        distances_sorted.reverse()     

        iters = range(len(rankedsolutions))
        plt.plot(iters, distances_sorted)
        return rankedsolutions[0][1]


def _check_for_laminate(lpdatabase, xiD, criteria):
    xiD_database = lpdatabase.keys()
    xiD = np.array(xiD)

    if len(xiD_database) == 0:
        return True

    for xiD_db in xiD_database:
        xiD_db = np.array(xiD_db)
        dist = calc_dist(xiD, xiD_db)
        if dist < criteria:
            return False # a close stack already exists
    return True
    


def generate_database_brute_force(n_plies, dist_criteria=0.05):
    '''
    Given number of plies and filtering rules, returns a new permutation.
    '''
    n = round((n_plies+0.1)/2)
    odd = isodd(n_plies)

    permutations = _layup_permutations_product(n)
    dist = 9999.9
    mat = OrthotropicMaterial(1000, 1000, 0.3, 500, 0.1) #Dummy

    database = {}

    niterations = 0
    for sequence in permutations:
        niterations += 1
        if (_is_balanced(sequence)
                and _contiguity(sequence)
                and _10p_rule(sequence)
                ):
            
            stack = convert_sequence_to_symmetric_laminate(sequence, isodd=odd)
            lam = LaminateProperty(stack, mat)
            xiD = np.array([lam.xiD[0], lam.xiD[2]])
            
            not_in_database = _check_for_laminate(database, xiD, criteria=dist_criteria)
            if not_in_database:
                xiDt = (xiD[0], xiD[1])
                database[xiDt] = stack
    return database



def generate_database_brute_force(n_plies, dist_criteria=0.05):
    '''
    Given number of plies and filtering rules, returns a new permutation.

    24 plies takes 1 hour. Not efficient to generate laminate with more plies
    '''
    n = round((n_plies+0.1)/2)
    odd = isodd(n_plies)

    permutations = _layup_permutations_product(n)
    dist = 9999.9
    mat = OrthotropicMaterial(1000, 1000, 0.3, 500, 0.1) #Dummy

    database = {}

    niterations = 0
    for sequence in permutations:
        niterations += 1
        if (_is_balanced(sequence)
                and _contiguity(sequence)
                and _10p_rule(sequence)
                ):
            
            stack = convert_sequence_to_symmetric_laminate(sequence, isodd=odd)
            lam = LaminateProperty(stack, mat)
            xiD = np.array([lam.xiD[0], lam.xiD[2]])
            
            not_in_database = _check_for_laminate(database, xiD, criteria=dist_criteria)
            if not_in_database:
                xiDt = (xiD[0], xiD[1])
                database[xiDt] = stack
    return database


def generate_database_random(n_plies, dist_criteria=0.05, n_generation=50000):
    '''
    Given number of plies and filtering rules, returns a new permutation.

    24 plies takes 1 hour. Not efficient to generate laminate with more plies
    '''

    dist = 9999.9
    mat = OrthotropicMaterial(1000, 1000, 0.3, 500, 0.1) #Dummy

    database = {}

    niterations = 0
    for _ in range(n_generation):
        niterations += 1
        stack = generate_laminate_sequences(n_plies, 1)[0]
        lam = LaminateProperty(stack, mat)
        xiD = np.array([lam.xiD[0], lam.xiD[2]])
        
        not_in_database = _check_for_laminate(database, xiD, criteria=dist_criteria)
        if not_in_database:
            xiDt = (xiD[0], xiD[1])
            database[xiDt] = stack
    return database





if __name__ == '__main__':

    import time


    # Brute force ==============
    # for nplies in range(10, 201):
    #     ti = time.time()
    #     file_name = f'db_{nplies}_plies.py'
    #     database = generate_database_brute_force(nplies)
    #     txt = f'db_{nplies}_plies = '
    #     txt += str(database)

    #     with open(file_name, 'w') as f:
    #         f.write(txt)


    # Random
    nplies = list(range(143, 166)) + list(range(118, 128)) + list(range(75, 91))
    nplies.sort(reverse=True)

    for n in nplies:       
        ti = time.time()
        file_name = f'db_{n}_plies.py'
        database = generate_database_random(n, dist_criteria=0.05, n_generation=60000)
        txt = f'db_{n}_plies = '
        txt += str(database)

        with open(file_name, 'w') as f:
            f.write(txt)

        print(f' time is {time.time()-ti}')

