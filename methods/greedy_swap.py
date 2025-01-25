import os
import pickle
import time

import numpy as np

from methods.swap_solver import SwapSolver
from results import PMPSolution
from utils import get_cost


class GreedySwapSolver(SwapSolver):
    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):
        start = time.time()
        best_sol = None

        mask = np.ones(city_pop.numel(), dtype=np.bool)
        mask[facility_list] = 0
        swaps = []

        for _ in range(reloc_step):
            fac_in_indices = np.where(mask == 1)[0]
            min_cost = get_cost(facility_list, distance_m, city_pop)
            best_action = None

            for i, fac_out in enumerate(facility_list):
                for fac_in in fac_in_indices:
                    facility_list_ = facility_list.copy()
                    facility_list_[i] = fac_in
                    cost = get_cost(facility_list_, distance_m, city_pop)
                    if cost < min_cost:
                        min_cost = cost
                        best_action = (fac_out, fac_in)
                    del facility_list_

            if best_action == None:
                break
            fac_out, fac_in = best_action
            facility_list[np.where(facility_list == fac_out)[0]] = fac_in
            mask[fac_in] = 0
            mask[fac_out] = 1
            swaps.append(best_action)
        best_sol = PMPSolution(facility_list, time.time() - start, min_cost)
        best_sol.swaps = swaps
        return best_sol


def run_greedy_swap(dataset, save_path, swap_num, init_num, **kwargs):
    name = f"greedy_swap_{init_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GreedySwapSolver(None)
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num, init_num)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path 


def run_greedy_swap_reloc(dataset, save_path, reloc_coef, **kwargs):
    name = "greedy_swap"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GreedySwapSolver(None)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(
                city_pop, p, distance_m, facility_list, int(reloc_coef * p)
            )
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path
