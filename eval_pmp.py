from dataset import GraphDataset
from methods.greedy_swap import run_greedy_swap
from methods.gurobi import run_gurobi
from methods.maranzana import run_maranzana
from methods.ppo_swap import run_ppo_swap
from methods.random_swap import run_random_swap
from methods.sa import run_SA
from results import save_avg, save_pmp_results
from utils import get_config


def run_pmp_graph(config):
    data_path = config['data_path']
    dataset = GraphDataset(data_path, "range(5, 41, 5)")
    save_path = f"{data_path}/result_pmp/"

    res_list = {}

    baseline = "Gurobi"
    optimal_path = run_gurobi(dataset=dataset, save_path=save_path, OutputFlag=0, MIPGap=0)
    res_list[baseline] = optimal_path
    save_avg(optimal_path, dataset, optimal_path)

    for k, v in config["methods"].items():
        run_fn = eval(v["run_fn"])
        sol_path = run_fn(dataset=dataset, save_path=save_path, **v)
        res_list[k] = sol_path
        save_avg(sol_path, dataset, optimal_path)

    save_pmp_results(save_path, res_list, dataset.p)


if __name__ == "__main__":
    config = get_config(["-c", "config/eval_pmp.yaml"])
    run_pmp_graph(config)
