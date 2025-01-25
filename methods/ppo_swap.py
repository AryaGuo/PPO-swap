import os
import pickle
import time

import numpy as np
import torch
import torch_geometric.data as geom_data

from methods.swap_solver import SwapSolver
from results import PMPSolution
from train import PPOLightning
from utils import get_cost


class PPOSwapSolver(SwapSolver):
    def __init__(self, iter_num, ckpt, device):
        super().__init__(iter_num)
        self.model = torch.load(ckpt, map_location=device)
        self.model = (
            PPOLightning.load_from_checkpoint(ckpt, mode="test").float().to(device)
        )
        self.device = device
        self.warm_up()

    def warm_up(self):
        rand_state = {
            "mask": torch.randint(
                0, 2, (self.iter_num, 100), dtype=torch.bool, device=self.device
            ),
            "fac_data": geom_data.Batch.from_data_list(
                [
                    geom_data.Data(
                        x=torch.rand(100, 7, device=self.device),
                        edge_index=torch.randint(0, 100, (2, 100), device=self.device),
                        edge_attr=torch.rand(100, 1, device=self.device),
                    )
                    for _ in range(self.iter_num)
                ]
            ),
        }
        with torch.no_grad():
            self.model(rand_state)

    def _get_fac_data(
        self,
        city_pop,
        p,
        distance_m,
        facility_list,
        static_feat,
        road_net_data,
        mask,
    ):

        wdist = distance_m[facility_list] * city_pop
        point_indices = torch.argmin(distance_m[facility_list], 0)
        node_costs = wdist[point_indices, torch.arange(distance_m.shape[1])]
        total_cost = torch.sum(node_costs)

        fac_costs = torch.zeros(p, device=wdist.device)
        fac_pop = torch.zeros(p, device=city_pop.device)

        fac_costs.scatter_add_(0, point_indices, node_costs)
        fac_pop.scatter_add_(0, point_indices, city_pop)

        fac_feat = torch.cat(
            (
                # fac_pop.reshape(-1, 1) / torch.sum(city_pop),
                # fac_costs.reshape(-1, 1) / total_cost,
                fac_pop.reshape(-1, 1) / torch.max(fac_pop),
                fac_costs.reshape(-1, 1) / torch.max(fac_costs),
            ),
            axis=1,
        )
        node_fac_feat = torch.zeros(
            (city_pop.shape[0], fac_feat.shape[1]), device=self.device
        )
        node_fac_feat[facility_list] = fac_feat

        node_feat = torch.cat(
            (
                static_feat,
                mask.reshape(-1, 1),
                # node_costs.reshape(-1, 1) / total_cost,
                node_costs.reshape(-1, 1) / torch.max(node_costs),
                node_fac_feat,
            ),
            axis=1,
        )

        fac_data = geom_data.Data(
            x=node_feat,
            edge_index=road_net_data.edge_index,
            edge_attr=road_net_data.edge_attr,
        )

        return fac_data, total_cost

    def solve_reloc(self, city_pop, p, distance_m, facility_list, reloc_step, **kwargs):

        start = time.time()
        best_sol = None
        city_pop = city_pop.to(self.device)
        distance_m = distance_m.to(self.device)
        coordinates = kwargs["coordinates"].to(self.device)
        road_net_data = kwargs["road_net_data"].to(self.device)
        coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
            torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
        )
        static_feat = torch.cat(
            # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
            (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop)),
            axis=1
        )
        facility_lists = np.tile(facility_list, (self.iter_num, 1))
        masks = torch.ones(
            (self.iter_num, city_pop.shape[0]), dtype=torch.bool, device=self.device
        )
        masks[:, facility_list] = 0

        for j in range(reloc_step):
            fac_data_list = []
            for i in range(self.iter_num):
                fac_data, cost = self._get_fac_data(
                    city_pop,
                    p,
                    distance_m,
                    facility_lists[i],
                    static_feat,
                    road_net_data,
                    masks[i],
                )
                fac_data_list.append(fac_data)
                if best_sol is None or cost < best_sol.cost:
                    best_sol = PMPSolution(facility_lists[i], np.nan, cost)

            state = {
                "mask": masks,
                "fac_data": geom_data.Batch.from_data_list(fac_data_list),
            }

            with torch.no_grad():
                action = self.model(state)[1].cpu().numpy()

            fac_out = action[:, 0]
            fac_in = action[:, 1]

            fac_out_index = np.where(facility_lists == fac_out[:, None])[1]
            facility_lists[np.arange(self.iter_num), fac_out_index] = fac_in

            masks[np.arange(self.iter_num), fac_out] = True
            masks[np.arange(self.iter_num), fac_in] = False

        for i in range(self.iter_num):
            wdist = distance_m[facility_lists[i]] * city_pop
            point_indices = torch.argmin(distance_m[facility_lists[i]], 0)
            cost = torch.sum(wdist[point_indices, torch.arange(distance_m.shape[1])])
            assert(get_cost(facility_lists[i], distance_m, city_pop) == cost)
            if best_sol is None or cost < best_sol.cost:
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)

        best_sol.time = time.time() - start

        return best_sol




def run_ppo_swap(
    dataset, save_path, iter_num, swap_num, init_num, ckpt, device, **kwargs
):
    name = f'ppo_swap_{init_num}_{iter_num}_{swap_num}_{kwargs["name"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = PPOSwapSolver(iter_num, ckpt, device)
    for batch in dataset:
        city_id, city_pop, p, distance_m, coordinates, road_net_data = batch[:6]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(
                p,
                city_pop,
                distance_m,
                swap_num,
                init_num,
                coordinates=coordinates,
                road_net_data=road_net_data,
            )
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
            
    return sol_path

def run_ppo_swap_reloc(dataset, save_path, iter_num, ckpt, device, reloc_coef, **kwargs):
    name = f'ppo_swap_{iter_num}_{kwargs["name"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = PPOSwapSolver(iter_num, ckpt, device)
    for batch in dataset:
        (
            city_id,
            city_pop,
            p,
            distance_m,
            coordinates,
            road_net_data,
            facility_list,
        ) = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(
                city_pop,
                p,
                distance_m,
                facility_list,
                int(reloc_coef * p),
                coordinates=coordinates,
                road_net_data=road_net_data,
            )
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path
