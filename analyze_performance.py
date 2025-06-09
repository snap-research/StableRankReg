import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataloader import *
from torch_geometric import seed_everything
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm
from utils import ALL_DATASET_ATTRS


def process_warm_start():

    datasets = [
        "MovieLens1M",
        "Gowalla",
        "Yelp2018",
        "AmazonBook",
    ]

    seed = 123
    seed_everything(seed)
    models = ["MLP", "LGConv"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = ["BPR", "align"]

    colors = ["r", "b", "g", "y", "m"]
    shapes = ["o", "^"]

    for m, model in enumerate(models):
        for d, dataset_name in enumerate(datasets):
            for l, loss in enumerate(losses):
                print(dataset_name, model, loss)
                if loss == "BPR":
                    reg = "-1"
                else:
                    reg = "uniformity"

                if loss == "BPR":
                    warm_start_regs = [[], ["stable_rank"]]
                elif loss == "align":
                    warm_start_regs = [[], ["stable_rank"], ["uniformity"]]

                results_file_name = f"results_warmstart/{dataset_name}_{model}_{loss}_{reg}_all_results.json"
    
                try:
                    with open(results_file_name, "r") as f:
                        results_data = json.load(f)
                except:
                    print(f"Missing {results_file_name}")
                    continue

                ndcg_results = np.array(results_data["ndcg"])
                num_models = len(ndcg_results)
                warm_start_method = [
                    results_data["model_configs"][i]["warm_start_regs"]
                    for i in range(num_models)
                ]
                best_val_ndcg = np.array(
                    [
                        results_data["model_configs"][i]["best_val_ndcg"]
                        for i in range(num_models)
                    ]
                )
                all_epochs = np.array(
                    [
                        results_data["model_configs"][i]["last_epoch"]
                        for i in range(num_models)
                    ]
                )

                best_perf_per_ws = []
                epochs = []
                for wsr in warm_start_regs:

                    idx = [i for i, w in enumerate(warm_start_method) if w == wsr]
                    val_results_wsr = best_val_ndcg[idx]
                    best_idx = np.argmax(val_results_wsr)

                    test_results_wsr = ndcg_results[idx]
                    curr_epochs = all_epochs[idx]

                    best_test = test_results_wsr[best_idx]
                    best_epoch = curr_epochs[best_idx]

                    best_perf_per_ws.append(best_test)
                    epochs.append(best_epoch)

                print(best_perf_per_ws)
                print(epochs)
def main():

    process_warm_start()


if __name__ == "__main__":
    main()
