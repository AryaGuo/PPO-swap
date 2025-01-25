import os
import pickle
import time

from results import PMPSolution
from utils import DensitySampling, get_cost


class SwapSolver:
    def __init__(self, iter_num):
        self.iter_num = iter_num

    def solve_reloc(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self, p, city_pop, distance_m, swap_num, init_num, **kwargs):
        start = time.time()
        best_sol = None
        if swap_num is None:
            swap_num = p
        for _ in range(init_num):
            facility_list = DensitySampling(1).sample(city_pop, p)
            sol = self.solve_reloc(
                city_pop, p, distance_m, facility_list, reloc_step=swap_num, **kwargs
            )
            if best_sol is None or sol.cost < best_sol.cost:
                best_sol = sol
        best_sol.time = time.time() - start
        return best_sol


def run_original(dataset, save_path, **kwargs):
    name = "original"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, facility_list = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            cost = get_cost(facility_list, distance_m, city_pop)
            sol = PMPSolution(facility_list, 0, cost)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
    return sol_path
