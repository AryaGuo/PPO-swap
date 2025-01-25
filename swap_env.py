import torch
import torch_geometric.data as geom_data

from dataset import GraphImpDataset
from utils import DensitySampling


class SwapEnv:
    def __init__(
        self,
        data_path,
        episode_len=40,
    ):
        self._dataset = GraphImpDataset(data_path=data_path, fac_range="range(5, 41)")
        self._index_iter = iter(range(len(self._dataset)))
        self._index = None
        self._steps = None
        self._episode_len = episode_len

        self.city_pop = None
        self.p = None
        self.distance_m = None
        self.road_net_data = None
        self.total_cost = None
        self.total_pop = None
        self.init_cost = None
        self.static_feat = None

        self.facility_list = None
        self.mask = None

    def _get_fac_data(self):
        wdist = self.distance_m[self.facility_list] * self.city_pop
        point_indices = torch.argmin(self.distance_m[self.facility_list], 0)
        node_costs = wdist[point_indices, torch.arange(self.distance_m.shape[1])]
        self.total_cost = torch.sum(node_costs)

        fac_costs = torch.zeros(self.p, device=wdist.device)
        fac_pop = torch.zeros(self.p, device=self.city_pop.device)

        fac_costs.scatter_add_(0, point_indices, node_costs)
        fac_pop.scatter_add_(0, point_indices, self.city_pop)

        fac_feat = torch.cat(
            (
                fac_pop.reshape(-1, 1) / torch.max(fac_pop),
                fac_costs.reshape(-1, 1) / torch.max(fac_costs),
            ),
            axis=1,
        )
        node_fac_feat = torch.zeros((self.city_pop.shape[0], fac_feat.shape[1]))
        node_fac_feat[self.facility_list] = fac_feat

        node_feat = torch.cat(
            (
                self.static_feat,
                self.mask.reshape(-1, 1),
                node_costs.reshape(-1, 1) / torch.max(node_costs),
                node_fac_feat,
            ),
            axis=1,
        )

        fac_data = geom_data.Data(
            x=node_feat,
            edge_index=self.road_net_data.edge_index,
            edge_attr=self.road_net_data.edge_attr,
        )

        return fac_data

    def _get_obs(self):
        return {"mask": self.mask, "fac_data": self._get_fac_data()}

    def _get_info(self):
        return {"cost": self.total_cost}

    def reset(self):
        try:
            self._index = next(self._index_iter)
        except StopIteration:
            self._index_iter = iter(range(len(self._dataset)))
            self._index = next(self._index_iter)

        (
            _,
            self.city_pop,
            self.p,
            self.distance_m,
            self.coordinates,
            self.road_net_data,
            self.facility_list,
        ) = self._dataset[self._index]

        self.total_pop = torch.sum(self.city_pop)
        self.static_feat = torch.cat(
            (
                self.coordinates,
                self.city_pop.reshape(-1, 1) / torch.max(self.city_pop),
            ),
            axis=1,
        )

        self._steps = 0
        self.facility_list = DensitySampling(1).sample(self.city_pop, self.p)
        self.mask = torch.ones(self.city_pop.shape[0], dtype=torch.bool)
        self.mask[self.facility_list] = 0

        observation = self._get_obs()
        info = self._get_info()
        self.init_cost = self.total_cost

        return observation, info

    def step(self, action):
        pre_cost = self.total_cost

        fac_out, fac_in = action
        if fac_out == fac_in:
            done = True
            return self._get_obs(), 0.0, done, False, self._get_info()

        assert self.mask[fac_out] == 0
        assert self.mask[fac_in] == 1

        self.facility_list[self.facility_list == fac_out] = fac_in
        self.mask[fac_out] = 1
        self.mask[fac_in] = 0

        self._steps += 1
        truncated = self._steps == self._episode_len
        done = False
        observation = self._get_obs()
        info = self._get_info()
        reward = (pre_cost - self.total_cost) / self.init_cost

        return observation, reward, done, truncated, info
