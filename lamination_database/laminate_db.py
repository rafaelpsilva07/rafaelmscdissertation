import numpy as np

from db_10_plies import db_10_plies
from db_11_plies import db_11_plies
from db_12_plies import db_12_plies
from db_13_plies import db_13_plies
from db_14_plies import db_14_plies
from db_15_plies import db_15_plies
from db_16_plies import db_16_plies
from db_17_plies import db_17_plies
from db_18_plies import db_18_plies
from db_19_plies import db_19_plies
from db_20_plies import db_20_plies
from db_21_plies import db_21_plies
from db_22_plies import db_22_plies
#from db_23_plies import db_23_plies
#from db_24_plies import db_24_plies


def calc_dist(p1, p2):
    return np.linalg.norm(p1-p2)


def get_best_laminate(xiD, nplies):
    xiD = np.array(xiD)
    db_plies = eval(f'db_{nplies}_plies')

    available_lp = list(db_plies.keys())
    dist = 9999.9
    stack = []

    for lp in available_lp:
        lp_array = np.array(lp)
        cur_dist = calc_dist(lp_array, xiD)
        if cur_dist < dist:
            stack = db_plies[lp]
            dist = cur_dist
    return stack



