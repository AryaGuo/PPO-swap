import os
import pickle

import networkx as nx
import numpy as np
import torch
import torch_geometric.data as geom_data
from torch.utils.data import Dataset


def preprocess_graph(graph, distance_m):
    # coordinates normalized to [0,1]^2; edges and dist are scaled to [0, 1]
    # city_pop are not scaled (for computing obj.); solvers should normalize by themselves

    if len(nx.get_node_attributes(graph, "pos")) > 0:
        coordinates = torch.Tensor(np.array(list(nx.get_node_attributes(graph, "pos").values())))
    else:
        node_x = torch.Tensor(np.array(list(nx.get_node_attributes(graph, "x").values())))
        node_y = torch.Tensor(np.array(list(nx.get_node_attributes(graph, "y").values())))
        coordinates = torch.stack([node_x, node_y], 1)

    city_pop = torch.Tensor(list(nx.get_node_attributes(graph, "pop").values()))

    scale = max(torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0])
    coordinates = (coordinates - torch.min(coordinates, 0)[0]) / scale
    
    edges = list(graph.edges(data="length"))
    edge_index = torch.LongTensor([(u, v) for u, v, _ in edges]).T  # [2, m]
    edge_attr = torch.Tensor([e[-1] for e in edges])

    edge_attr = edge_attr.reshape(-1, 1) / edge_attr.max()  # [m, 1]
    road_net_data = geom_data.Data(edge_index=edge_index, edge_attr=edge_attr)
    distance_m = distance_m / distance_m.max()

    return (coordinates, road_net_data, distance_m, city_pop)


class GraphDataset(Dataset):
    def __init__(self, data_path: str, fac_range):
        super().__init__()

        self.p = list(eval(fac_range))
        self.data_path = data_path
        self.city_num = int(data_path.rstrip("/").split("_")[-1])
        self.city_pops = []
        self.distance_m = []
        self.coordinates = []
        self.road_net_data = []

        for i in range(self.city_num):
            G = pickle.load(open(f"{data_path}/{i}/graph.pkl", "rb"))
            distance_m_i = torch.Tensor(pickle.load(open(f"{data_path}/{i}/distance_m.pkl", "rb"))) 
            (coordinates, road_net_data, distance_m_i, city_pop) = preprocess_graph(G, distance_m_i)

            self.city_pops.append(city_pop)
            self.distance_m.append(distance_m_i)
            self.coordinates.append(coordinates)
            self.road_net_data.append(road_net_data)

    def __len__(self):
        return self.city_num * len(self.p)

    def __getitem__(self, index):
        city_id = index // len(self.p)
        p = self.p[index % len(self.p)]
        return (
            city_id,
            self.city_pops[city_id].clone(),
            p,
            self.distance_m[city_id].clone(),
            self.coordinates[city_id].clone(),
            self.road_net_data[city_id].clone(),
        )


class GraphImpDataset(GraphDataset):
    def __init__(self, data_path: str, fac_range):
        super().__init__(data_path, fac_range)
        self.init_dir = f"{self.data_path}/init/"
        os.makedirs(self.init_dir, exist_ok=True)

    def __getitem__(self, index):
        (
            city_id,
            city_pop,
            p,
            distance_m,
            coordinates,
            road_net_data,
        ) = super().__getitem__(index)

        if os.path.isfile(f"{self.init_dir}/{city_id}_{p}.pkl"):
            init_facility = pickle.load(
                open(f"{self.init_dir}/{city_id}_{p}.pkl", "rb")
            )
        else:
            init_facility = np.random.choice(
                len(self.city_pops[city_id]), size=p, replace=False
            )
            pickle.dump(
                init_facility,
                open(f"{self.init_dir}/{city_id}_{p}.pkl", "wb"),
                pickle.HIGHEST_PROTOCOL,
            )
        return (
            city_id,
            city_pop,
            p,
            distance_m,
            coordinates,
            road_net_data,
            init_facility.copy(),
        )

