import csv
import math
import pickle

import numpy as np
import torch

from utils import get_cost


class PMPSolution:
    def __init__(self, facility_list, time, cost=None):
        self.time = time

        if isinstance(facility_list, torch.Tensor):
            self.facility_list = facility_list.detach().cpu().numpy()
        elif isinstance(facility_list, np.ndarray):
            self.facility_list = facility_list
        else:
            self.facility_list = np.asarray(facility_list)

        if isinstance(cost, torch.Tensor):
            self.cost = cost.item()
        elif isinstance(cost, np.ndarray):
            self.cost = cost.item()
        else:
            self.cost = cost

    def eval(self, city_pop, distance_m):
        self.cost = get_cost(self.facility_list, distance_m, city_pop).item()


def save_avg(sol_path, dataset, optmial_path, reloc=None):
    costs, rtimes, gaps, imps = {}, {}, {}, {}

    for batch in dataset:
        city_id, p = batch[0], batch[2]
        sol = pickle.load(open(sol_path + f"/{city_id}_{p}.pkl", "rb"))
        opt_sol = pickle.load(open(optmial_path + f"/{city_id}_{p}.pkl", "rb"))

        costs.setdefault(p, []).append(sol.cost)
        rtimes.setdefault(p, []).append(sol.time)
        gaps.setdefault(p, []).append((sol.cost - opt_sol.cost) / opt_sol.cost)
        if reloc:
            ori_sol = pickle.load(open(reloc + f"/{city_id}_{p}.pkl", "rb"))
            imps.setdefault(p, []).append((ori_sol.cost - sol.cost) / ori_sol.cost)

    avg_cost, avg_rtime, avg_gap, avg_imp = {}, {}, {}, {}
    std_cost, std_rtime, std_gap, std_imp = {}, {}, {}, {}

    for k, v in costs.items():
        avg_cost[k] = sum(v) / len(v)
        std_cost[k] = math.sqrt(sum((x - avg_cost[k]) ** 2 for x in v) / len(v))
    pickle.dump(avg_cost, open(sol_path + "/avg_cost.pkl", "wb"))
    pickle.dump(std_cost, open(sol_path + "/std_cost.pkl", "wb"))

    for k, v in rtimes.items():
        avg_rtime[k] = sum(v) / len(v)
        std_rtime[k] = math.sqrt(sum((x - avg_rtime[k]) ** 2 for x in v) / len(v))
    pickle.dump(avg_rtime, open(sol_path + "/avg_time.pkl", "wb"))
    pickle.dump(std_rtime, open(sol_path + "/std_time.pkl", "wb"))

    for k, v in gaps.items():
        avg_gap[k] = sum(v) / len(v)
        std_gap[k] = math.sqrt(sum((x - avg_gap[k]) ** 2 for x in v) / len(v))

    pickle.dump(avg_gap, open(sol_path + "/avg_gap.pkl", "wb"))
    pickle.dump(std_gap, open(sol_path + "/std_gap.pkl", "wb"))

    if reloc:
        for k, v in imps.items():
            avg_imp[k] = sum(v) / len(v)
            std_imp[k] = math.sqrt(sum((x - avg_imp[k]) ** 2 for x in v) / len(v))
        pickle.dump(avg_imp, open(sol_path + "/avg_imp.pkl", "wb"))
        pickle.dump(std_imp, open(sol_path + "/std_imp.pkl", "wb"))


def calculate_average(data):
    return sum(data.values()) / len(data.values())


def calculate_gap(y, optimal, facility_range):
    gap = {}
    for k in facility_range:
        gap[k] = (
            0 if math.isclose(y[k], optimal[k]) else ((y[k] - optimal[k]) / optimal[k])
        )
    return gap


def calculate_imp(y, original, facility_range):
    imp = {}
    for k in facility_range:
        imp[k] = (
            0
            if math.isclose(y[k], original[k])
            else ((original[k] - y[k]) / original[k])
        )
    return imp


def write_to_csv(filename, header, data):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)


def save_pmp_results(save_path, res_list, facility_range):
    cost_data = []
    run_time_data = []
    optimality_gap_data = []
    overall_data = []

    overall_header = ["facility number", "average", ""]
    for p in facility_range:
        overall_header.extend([f"{p}", ""])
    second_line = ["Methods", "Gap (%)", "Time (s)"]
    for p in facility_range:
        second_line.extend(["Gap (%)", "Time (s)"])
    overall_data.append(second_line)

    for name, fin in res_list.items():
        avg_cost = pickle.load(open(fin + "/avg_cost.pkl", "rb"))
        std_cost = pickle.load(open(fin + "/std_cost.pkl", "rb"))
        avg_time = pickle.load(open(fin + "/avg_time.pkl", "rb"))
        std_time = pickle.load(open(fin + "/std_time.pkl", "rb"))
        avg_gap = pickle.load(open(fin + "/avg_gap.pkl", "rb"))
        std_gap = pickle.load(open(fin + "/std_gap.pkl", "rb"))

        cost_avg_all = calculate_average(avg_cost)
        time_avg_all = calculate_average(avg_time)
        gap_avg_all = calculate_average(avg_gap)

        cost_row = [name, f"{cost_avg_all:.3g}"] + [
            f"{avg_cost[p]:.3g} $\pm$ {std_cost[p]:.3g}" for p in facility_range
        ]
        cost_data.append(cost_row)

        run_time_row = [name, f"{time_avg_all:.2f}"] + [
            f"{avg_time[p]:.2f} $\pm$ {std_time[p]:.2f}" for p in facility_range
        ]
        run_time_data.append(run_time_row)

        gap_row = [name, f"{gap_avg_all*100:.2f}"] + [
            f"{avg_gap[p]*100:.2f} $\pm$ {std_gap[p]*100:.2f}" for p in facility_range
        ]
        optimality_gap_data.append(gap_row)

        overall_row = [
            name,
            f"{gap_avg_all*100:.2f}",
            f"{time_avg_all:.2f}",
        ]
        for p in facility_range:
            overall_row.extend(
                [
                    f"{avg_gap[p]*100:.2f} $\pm$ {std_gap[p]*100:.2f}",
                    f"{avg_time[p]:.2f} $\pm$ {std_time[p]:.2f}",
                ]
            )
        overall_data.append(overall_row)

    write_to_csv(
        save_path + "/obj.csv",
        ["facility number", "average"] + [f"{p}" for p in facility_range],
        cost_data,
    )
    write_to_csv(
        save_path + "/run_time.csv",
        ["facility number", "average"] + [f"{p}" for p in facility_range],
        run_time_data,
    )
    write_to_csv(
        save_path + "/optimality_gap.csv",
        ["facility number", "average"] + [f"{p}" for p in facility_range],
        optimality_gap_data,
    )

    write_to_csv(save_path + "/overall.csv", overall_header, overall_data)


def save_frp_results(save_path, res_list, facility_range):
    cost_data = []
    run_time_data = []
    imp_data = []
    optimality_gap_data = []
    overall_data = []

    overall_header = ["facility number", "average", "", ""]
    for p in facility_range:
        overall_header.extend([f"{p}", "", ""])
    second_line = ["Methods", "$Q$ (\%)", "Gap (%)", "Time (s)"]
    for p in facility_range:
        second_line.extend(["$Q$ (\%)", "Gap (%)", "Time (s)"])
    overall_data.append(second_line)

    for name, fin in res_list.items():
        avg_cost = pickle.load(open(fin + "/avg_cost.pkl", "rb"))
        std_cost = pickle.load(open(fin + "/std_cost.pkl", "rb"))
        avg_time = pickle.load(open(fin + "/avg_time.pkl", "rb"))
        std_time = pickle.load(open(fin + "/std_time.pkl", "rb"))
        avg_imp = pickle.load(open(fin + "/avg_imp.pkl", "rb"))
        std_imp = pickle.load(open(fin + "/std_imp.pkl", "rb"))
        avg_gap = pickle.load(open(fin + "/avg_gap.pkl", "rb"))
        std_gap = pickle.load(open(fin + "/std_gap.pkl", "rb"))

        cost_avg_all = calculate_average(avg_cost)
        time_avg_all = calculate_average(avg_time)
        gap_avg_all = calculate_average(avg_gap)
        imp_avg_all = calculate_average(avg_imp)

        cost_row = [name, f"{cost_avg_all:.3g}"] + [
            f"{avg_cost[p]:.3g} $\pm$ {std_cost[p]:.3g}" for p in facility_range
        ]
        cost_data.append(cost_row)

        run_time_row = [name, f"{time_avg_all:.2f}"] + [
            f"{avg_time[p]:.2f} $\pm$ {std_time[p]:.2f}" for p in facility_range
        ]
        run_time_data.append(run_time_row)

        gap_row = [name, f"{gap_avg_all*100:.2f}"] + [
            f"{avg_gap[p]*100:.2f} $\pm$ {std_gap[p]*100:.2f}" for p in facility_range
        ]
        optimality_gap_data.append(gap_row)

        imp_row = [name, f"{imp_avg_all*100:.2f}"] + [
            f"{avg_imp[p]*100:.2f} $\pm$ {std_imp[p]*100:.2f}" for p in facility_range
        ]
        imp_data.append(imp_row)

        overall_row = [
            name,
            f"{imp_avg_all*100:.2f}",
            f"{gap_avg_all*100:.2f}",
            f"{time_avg_all:.2f}",
        ]
        for p in facility_range:
            overall_row.extend(
                [
                    f"{avg_imp[p]*100:.2f} $\pm$ {std_imp[p]*100:.2f}",
                    f"{avg_gap[p]*100:.2f} $\pm$ {std_gap[p]*100:.2f}",
                    f"{avg_time[p]:.2f} $\pm$ {std_time[p]:.2f}",
                ]
            )
        overall_data.append(overall_row)

    write_to_csv(
        save_path + "/obj.csv",
        ["facility number", "average"] + [f"{p}" for p in facility_range],
        cost_data,
    )
    write_to_csv(
        save_path + "/run_time.csv",
        ["facility number", "average"] + [f"{p}" for p in facility_range],
        run_time_data,
    )
    write_to_csv(
        save_path + "/improvement.csv",
        ["facility number", "average"] + [f"{p}" for p in facility_range],
        imp_data,
    )
    write_to_csv(
        save_path + "/optimality_gap.csv",
        ["facility number", "average"] + [f"{p}" for p in facility_range],
        optimality_gap_data,
    )

    write_to_csv(save_path + "/overall.csv", overall_header, overall_data)
