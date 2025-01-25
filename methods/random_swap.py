import os
import pickle
import time

import numpy as np

from methods.swap_solver import SwapSolver
from results import PMPSolution
from utils import get_cost


class RandomSwapSolver(SwapSolver):
    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):
        start = time.time()
        best_sol = None

        facility_list_ = facility_list.copy()

        for i in range(self.iter_num):
            facility_list = facility_list_.copy()
            mask = np.ones(np.prod(city_pop.shape), dtype=np.bool)
            mask[facility_list] = 0
            swaps = []

            for j in range(reloc_step):
                fac_in_indices = np.where(mask == 1)[0]

                fac_out_idx = np.random.choice(range(len(facility_list)))
                fac_out = facility_list[fac_out_idx]
                fac_in = np.random.choice(fac_in_indices)
                facility_list[fac_out_idx] = fac_in
                cost = get_cost(facility_list, distance_m, city_pop)

                mask[fac_in] = 0
                mask[fac_out] = 1

                swaps.append((fac_out, fac_in))
                if best_sol is None or cost < best_sol.cost:
                    best_sol = PMPSolution(facility_list, np.nan, cost)
                    best_sol.swaps = swaps

            cost = get_cost(facility_list, distance_m, city_pop)

        best_sol.time = time.time() - start
        return best_sol


def run_random_swap(dataset, save_path, iter_num, swap_num, init_num, **kwargs):
    name = f"random_swap_{init_num}_{iter_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = RandomSwapSolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num, init_num)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path


def run_random_swap_reloc(dataset, save_path, iter_num, reloc_coef, **kwargs):
    name = f"random_swap_{iter_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = RandomSwapSolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(
                city_pop, p, distance_m, facility_list, int(reloc_coef * p)
            )
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path
