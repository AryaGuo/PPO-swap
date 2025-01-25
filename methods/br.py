import math
import os
import pickle
import time

import numpy as np

from results import PMPSolution
from utils import get_cost


class BestResponse:
    def __init__(self, iter_num):
        self.iter_num = iter_num

    def solve(self, city_pop, p, distance_m, facility_list, reloc_step):
        def get_best_move(idx, fac_in_indices, facility_list):
            fac_out = facility_list[idx]
            min_cost = get_cost(facility_list, distance_m, city_pop)
            best_action = fac_out
            for fac_in in fac_in_indices:
                facility_list_ = facility_list.copy()
                facility_list_[idx] = fac_in
                cost = get_cost(facility_list_, distance_m, city_pop)
                if cost < min_cost:
                    min_cost = cost
                    best_action = fac_in
                del facility_list_
            return best_action

        start = time.time()
        best_sol = None

        fl_ = facility_list.copy()

        for i in range(self.iter_num):
            mask = np.ones(city_pop.numel(), dtype=np.bool)
            facility_list = fl_.copy()
            mask[facility_list] = 0
            steps = reloc_step

            while(steps > 0):
                flag = False
                x = list(range(p))
                np.random.shuffle(x)
                for i in x:
                    fac_out = facility_list[i]
                    fac_in = get_best_move(i, np.where(mask == 1)[0], facility_list)
                    if mask[fac_in] and fac_out != fac_in:
                        flag = True
                        facility_list[i] = fac_in
                        mask[fac_in] = 0
                        mask[fac_out] = 1
                        steps -= 1
                        if steps == 0:
                            break
                if not flag:
                    break

            cost = get_cost(facility_list, distance_m, city_pop)
            if best_sol is None or cost < best_sol.cost:
                best_sol = PMPSolution(facility_list, time.time() - start, cost)

        best_sol.time = time.time() - start
        return best_sol


def run_br(dataset, save_path, iter_num, reloc_coef, **kwargs):
    name = f"BestResponse_{iter_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    rs = BestResponse(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = rs.solve(city_pop, p, distance_m, facility_list, int(reloc_coef * p))
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
    return sol_path
