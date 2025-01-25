import os
import pickle
import random
import time

import numpy as np

from results import PMPSolution
from utils import get_cost


class SimulatedAnnealingSolver:
    def __init__(self, initial_temp, cooling_rate):
        self.p = None
        self.facility_list = None
        self.candidates = None
        self.cost = None
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def step(self, city_pop, distance_m):
        neighbor = self.facility_list.copy()
        fac_out = random.choice(self.facility_list)
        fac_in = random.choice(self.candidates)
        neighbor[neighbor.index(fac_out)] = fac_in
        self.candidates.remove(fac_in)
        self.candidates.append(fac_out)

        neighbor_cost = get_cost(neighbor, distance_m, city_pop)
        delta_cost = neighbor_cost - self.cost

        if delta_cost < 0 or random.random() < np.exp(-delta_cost / self.current_temp):
            self.facility_list = neighbor
            self.cost = neighbor_cost

        self.current_temp *= self.cooling_rate

    def solve(self, p, city_pop, distance_m, iter_num):
        start = time.time()
        self.p = p
        self.facility_list = random.sample(range(city_pop.numel()), p)
        self.candidates = list(range(city_pop.numel()))
        for i in self.facility_list:
            self.candidates.remove(i)
        self.cost = get_cost(self.facility_list, distance_m, city_pop)
        self.current_temp = self.initial_temp

        for _ in range(iter_num):
            self.step(city_pop, distance_m)

        sol = PMPSolution(self.facility_list, time.time() - start)
        sol.eval(city_pop, distance_m)
        return sol


def run_SA(dataset, save_path, iter_num, initial_temp, cooling_rate, **kwargs):
    name = f"SA_{iter_num}_{initial_temp}_{cooling_rate}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = SimulatedAnnealingSolver(initial_temp, cooling_rate)
    for batch in dataset:
        city_id, city_pop, p, distance_m = batch[:4]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, iter_num)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
            
    return sol_path
