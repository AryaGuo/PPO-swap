import math
import os
import pickle
import time

import numpy as np

from results import PMPSolution
from utils import get_cost


class FR2FP:
    def __init__(self):
        pass

    def solve(self, city_pop, p, distance_m, facility_list, reloc_step=None):
        start = time.time()
        best_sol = None
        if reloc_step is None:
            reloc_step = int(p * 0.5)

        F = facility_list.tolist()
        C = [i for i in range(city_pop.numel()) if i not in F]
        mask = np.ones(city_pop.numel(), dtype=np.bool)
        facility_list = []
        swaps = []
        num = 0

        for _ in range(p):
            fac_in_indices = np.where(mask == 1)[0]
            min_cost = math.inf
            best_in = None

            for fac_in in fac_in_indices:
                facility_list.append(fac_in)
                cost = get_cost(facility_list, distance_m, city_pop)
                if cost < min_cost:
                    min_cost = cost
                    best_in = fac_in
                facility_list.pop()
            facility_list.append(best_in)
            mask[best_in] = 0

            if best_in in C:
                num += 1
                if num == reloc_step:
                    mask[C] = 0  # cannot choose from C

        swap_outs = list(set(F) - set(facility_list))
        swap_ins = list(set(C) & set(facility_list))
        swaps = [(swap_out, swap_in) for swap_out, swap_in in zip(swap_outs, swap_ins)]

        best_sol = PMPSolution(facility_list, time.time() - start, min_cost)
        best_sol.swaps = swaps
        return best_sol


def run_FR2FP(dataset, save_path, reloc_coef, **kwargs):
    name = f"FR2FP"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    rs = FR2FP()
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = rs.solve(city_pop, p, distance_m, facility_list, int(reloc_coef * p))
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
    return sol_path
