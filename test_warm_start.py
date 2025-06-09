import argparse
import json
import os
import sys
from datetime import datetime
from typing import overload

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataloader import Data
from evaluation_metrics import *
from models import RecModel
from torch_geometric import seed_everything
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm
from utils import *


class Tester:
    def __init__(self, model, model_type, device, dataset):

        self.model = model
        self.model.to(device)
        self.model_type = model_type
        self.device = device
        self.user_attr = dataset.user
        self.action_attr = dataset.action
        self.item_attr = dataset.item

    def eval_model(self, train_data, val_data, test_data, batch_size=-1, k=-1):

        with torch.no_grad():
            test_data.to(self.device)

            # get the train/val indices for interactions, will remove these from rec matrix
            train_label_idx = train_data[
                self.user_attr, self.action_attr, self.item_attr
            ]["link_index"]
            val_label_idx = val_data[self.user_attr, self.action_attr, self.item_attr][
                "link_index"
            ]

            # these are what we will use for performance
            test_label_idx = test_data[
                self.user_attr, self.action_attr, self.item_attr
            ]["link_index"]

            if self.model_type == "LGConv":
                x_user, x_item = self.model.get_embeddings(
                    test_data.x_dict,
                    test_data.edge_index_dict,
                )
            else:
                x_user, x_item = self.model.get_embeddings(test_data.x_dict)

            test_preds = torch.zeros((x_user.shape[0], x_item.shape[0]))
            # The full matrix mul doesnt fit into memory in many cases, thus will sometimes need to batch it
            num_users = x_user.shape[0]
            for user_start in torch.arange(0, num_users, step=batch_size):
                user_preds = torch.matmul(
                    x_user[user_start : user_start + batch_size, :], x_item.T
                )
                test_preds[user_start : user_start + batch_size, :] = user_preds

            # zero out vals from train and val
            sources = train_label_idx[0, :]
            targets = train_label_idx[1, :]
            test_preds[sources, targets] = -1 * np.inf

            # sources = val_label_idx[0, :]
            # targets = val_label_idx[1, :]
            # test_preds[sources, targets] = -1 * np.inf

            top_k_recs = torch.topk(test_preds, k=k, dim=1).indices
            sorted_label_idx = sort_edge_index(test_label_idx)

        return sorted_label_idx, top_k_recs


def main():

    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument("--dataset", type=str, default="MovieLens1M")
    parser.add_argument("--dataset_save_path", type=str, default="./datasets")

    # model parameters
    parser.add_argument(
        "--model", type=str, default="LGConv", choices=["LGConv", "MLP"]
    )
    parser.add_argument(
        "--model_save_path", type=str, default="./models_chkp_warmstart"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--loss", type=str, default="align")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--best_hyperparam", type=str2bool, default=False)
    parser.add_argument(
        "--reg_types",
        type=str,
        default="uniformity",
        help="pass delimited string of regularization terms to be parsed",
    )

    # evaluation parameters
    parser.add_argument("--k", type=int, default=20)

    args = parser.parse_args()

    # seed everything
    seed_everything(args.seed)

    ### GET TEST PERFORMANCE ###
    results_file = f"results_warmstart/{args.dataset}_{args.model}_{args.loss}_{args.reg_types}_all_results.json"
    # check if results file exists
    if os.path.isfile(results_file):
        print("Already Processed")
        sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_save_folder = args.dataset_save_path + "/" + args.dataset + "/"

    # Set up save folders
    if args.dataset == "Gowalla" or args.dataset == "Yelp2018":
        args.dataset_save_path = (
            args.dataset_save_path + "/" + args.dataset + "/data.pt"
        )
    else:
        args.dataset_save_path = args.dataset_save_path + "/" + args.dataset
    args.base_model_save_path = (
        args.model_save_path + "/" + args.dataset + "/" + args.model + "_" + args.loss
    )
 
    args.base_model_save_path += (
        "_" + args.reg_types.replace(",", "_") if args.reg_types != -1 else ""
    )

    # load saved model if it exist, if not exit
    # to figure out where the model is saved, we will load all the params files and parse them
    file_names, files = parse_param_files(
        args.base_model_save_path,
        loss_filter=args.loss,
        hidden_dim_filter=args.hidden_dim,
        seed_filter=args.seed,
        best_hyperparam=args.best_hyperparam,
    )

    if len(file_names) == 0:
        # there were no files for this dataset, model, seed combo, just exit
        print(f"No models for {args.model}, {args.dataset}, {args.seed}")
        sys.exit()

    # iterate through the trained models, set up associated dataloaders, and get testing metrics
    all_results = {}
    all_results["global_recall"] = []
    all_results["ndcg"] = []
    all_results["local_recall"] = []
    all_results["local_ndcg"] = []
    all_results["source_nodes"] = []
    all_results["test_degree"] = []
    all_results["train_loss"] = []
    all_results["val_ndcg"] = []
    all_results["model_configs"] = []
    all_results["user_embed"] = []
    all_results["item_embed"] = []
    all_results["aligns"] = []
    all_results["uniforms"] = []

    print(args)
    for file in file_names:

        model_configs = files[file]

        # unpack key info
        num_layers = model_configs["num_layers"]
        hid_dims = model_configs["hidden_dim"]

        # dataset object holds the additional info to index into data object
        dataset = Data(
            args.dataset,
            args.dataset_save_path,
            split_save_folder,
        )

        # only need test data here
        train_data, val_data, test_data = dataset.get_dataloaders(
            num_layers=num_layers, testing=True
        )

        num_users = train_data[dataset.user].num_nodes
        num_items = train_data[dataset.item].num_nodes
        dataset.data = train_data

        # sets up the model for us, including the encoder and decoder steps
        model = RecModel(
            dataset,
            num_users,
            num_items,
            args.model,
            hidden_dim=hid_dims,
            depth=num_layers,
            device=device,
        )

        # load model from save file
        model.load_state_dict(torch.load(file, map_location=device))

        tester = Tester(model, args.model, device, dataset)

        # get recommendation matrix
        sorted_label_idx, rec_matrix = tester.eval_model(
            train_data, val_data, test_data, batch_size=500, k=args.k
        )

        # recall and ndcg at k on recommendations
        (
            recall_score_at_k,
            user_recall_score_at_k,
            source_nodes,
            test_degrees,
        ) = compute_recall_at_k(rec_matrix, sorted_label_idx)

        (
            ndcg_at_k,
            user_ndcg_at_k,
        ) = compute_ndcg_at_k(rec_matrix, sorted_label_idx)

        # save the source nodes and user_recall_at_k
        # save all of this data for later computation
        all_results["global_recall"].append(recall_score_at_k)
        all_results["ndcg"].append(ndcg_at_k)
        all_results["local_recall"].append(user_recall_score_at_k.tolist())
        all_results["local_ndcg"].append(user_ndcg_at_k.tolist())
        all_results["source_nodes"].append(source_nodes.tolist())
        all_results["test_degree"].append(test_degrees.tolist())
        all_results["train_loss"].append(model_configs["train_loss"])
        all_results["val_ndcg"].append(model_configs["val_ndcg"])
        all_results["model_configs"].append(model_configs)

    with open(results_file, "w") as f:
        json.dump(all_results, f)
    print(ndcg_at_k)


if __name__ == "__main__":
    main()
