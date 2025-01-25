import argparse

import numpy as np
import torch
import yaml


def get_config(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="config/config.yaml",
    )
    args = parser.parse_args(args)

    with open(args.filename) as yml_file:
        try:
            config = yaml.safe_load(yml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def to_device(state: dict, device):
    if isinstance(state, dict):
        for k, v in state.items():
            state[k] = v.to(device)
    return state


def get_cost(facility_list, distance_m, city_pop):
    total_cost = torch.sum(
        (distance_m[facility_list] * city_pop.flatten())[
            torch.argmin(distance_m[facility_list], axis=0),
            torch.arange(distance_m.shape[1]),
        ]
    )
    return total_cost


class DensitySampling:
    def __init__(self, exp):
        self.exp = exp

    def sample(self, city_pop, p):
        density = np.reshape(np.array(city_pop**self.exp), -1)
        density = density / np.sum(density)
        facility_list = np.random.choice(
            city_pop.numel(), size=p, p=density, replace=False
        )
        return facility_list
