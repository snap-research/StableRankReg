import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import overload

import torch
import torch.nn.functional as F
from dataloader import Data
from evaluation_metrics import *
from losses import AlignmentLoss, BPRLoss, SSMLoss
from models import RecModel
from torch.optim import Adam
from torch_geometric import seed_everything
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm
from utils import ALL_DATASET_ATTRS, get_processed_params, str2bool
import time


class Trainer:
    def __init__(
        self,
        model_type,
        model,
        device,
        dataset,
        args,
        optimizer="Adam",
    ):

        self.model_type = model_type
        self.model = model.to(device)

        self.optimizer = eval(optimizer)(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.device = device

        self.user_attr = dataset.user
        self.action_attr = dataset.action
        self.item_attr = dataset.item

        self.loss = args.loss
        self.regs = args.regs
        self.gammas = args.gammas

        self.warm_start_regs = args.warm_start_regs
        self.warm_start_gammas = args.warm_start_gammas
        # if we pass a metric, this will take precedence over specifying an epoch 
        self.warm_start_metric = args.warm_start_metric
        if self.warm_start_metric:
            self.warm_start_epochs = args.epochs
        else: 
            self.warm_start_epochs = args.warm_start_epochs
        self.phase = 'warm'

        # this sets up the base loss function that will be used during training
        if args.loss is None or args.loss == "BPR":
            self.loss_func = BPRLoss()
            self.neg_ratio = args.neg_ratio
        elif args.loss == 'SSM':
            self.loss_func = SSMLoss()
            self.neg_ratio = args.neg_ratio
        elif args.loss == "align":
            self.loss_func = AlignmentLoss()
            self.neg_ratio = 0

    def forward_and_get_loss(self, data, edge_label_index, train=True, epoch=None):

        if self.model_type == "MLP":
            x_user, x_item = self.model.get_embeddings(
                data.x_dict,
                user_id=data[self.user_attr].n_id if train else None,
                item_id=data[self.item_attr].n_id if train else None,
            )
        else:
            x_user, x_item = self.model.get_embeddings(
                data.x_dict,
                edge_index_dict=data.edge_index_dict,
                user_id=data[self.user_attr].n_id if train else None,
                item_id=data[self.item_attr].n_id if train else None,
            )
        user_train = x_user[edge_label_index[0]]
        item_train = x_item[edge_label_index[1]]

        if (len(self.warm_start_regs) > 0) or (len(self.regs) > 0):
            # if using regs, need to compute the in-batch user/item sets
            unique_users = torch.unique(edge_label_index[0, :])
            unique_items = torch.unique(edge_label_index[1, :])
            x_user_batch = x_user[unique_users, :]
            x_item_batch = x_item[unique_items, :]
        else:
            x_user_batch = None
            x_item_batch = None

        # based on loss, we will use different forward functions
        if self.loss == "BPR" or self.loss == 'SSM':
            # for BPR, we will use both encoder and decoder for pair-wise loss on scores
            preds = (user_train * item_train).sum(dim=-1)
            loss = self.loss_func(
                preds,
                embed_user=x_user_batch,
                embed_item=x_item_batch,
                neg_ratio=self.neg_ratio if train else 1,
                regs=self.regs if self.phase == 'tune' else self.warm_start_regs,
                gammas=self.gammas if self.phase == 'tune' else self.warm_start_gammas,
            )
        elif self.loss == "align":
            # index into embedding tables to get pair-wise embeddings
            loss = self.loss_func(
                user_train,
                item_train,
                embed_user=x_user_batch,
                embed_item=x_item_batch,
                regs=self.regs if self.phase == 'tune' else self.warm_start_regs,
                gammas=self.gammas if self.phase == 'tune' else self.warm_start_gammas,
            )

        return loss

    def forward_and_get_ndgc(
        self, data, train_edge_index, val_edge_index, k=20, batch_size=1000
    ):

        # get user and item embeddings
        if self.model_type == "MLP":
            x_user, x_item = self.model.get_embeddings(data.x_dict)
        else:
            x_user, x_item = self.model.get_embeddings(
                data.x_dict,
                data.edge_index_dict,
            )

        # The full matrix mul doesnt fit into memory in many cases, thus will sometimes need to batch it
        preds = torch.zeros((x_user.shape[0], x_item.shape[0]))

        # go through batches of users
        for user_start in torch.arange(0, x_user.shape[0], step=batch_size):
            # pairwise sim for batch of users and items
            user_preds = torch.matmul(
                x_user[user_start : user_start + batch_size, :], x_item.T
            )

            preds[user_start : user_start + batch_size, :] = user_preds

        # Mask out the train indices
        sources = train_edge_index[0, :]
        targets = train_edge_index[1, :]
        preds[sources, targets] = -1 * np.inf

        # get top k recommendations for the full rec matrix
        top_k_recs = torch.topk(preds, k=k, dim=1).indices
        (
            ndcg_at_k,
            user_ndcg_at_k,
        ) = compute_ndcg_at_k(top_k_recs, sort_edge_index(val_edge_index))

        return ndcg_at_k

    def train_model(
        self,
        train_loader,
        train_data,
        val_data,
        epochs=10,
        num_saves=30,
        early_stop_cutoff=10,
    ):

        train_loss = []
        val_ndcg = []

        # num saves tells us how many checkpoints to make
        if num_saves != 0:
            model_checkpoints = {}
            save_epoch_freq = epochs // num_saves
            # initial save
            model_checkpoints[-1] = copy.deepcopy(self.model)

        best_model = self.model
        best_val_ndcg = 0
        last_epoch = 0
        early_stop_counter = 0
        switch_over_epoch = 0

        times = []
        for e in range(epochs):
            total_loss = total_examples = 0
            # if we are choosing to switch via epoch number, switch here
            if not self.warm_start_metric and e > self.warm_start_epochs:
                self.phase = 'tune'

            t_start_epoch = time.time()

            ### Train step ###
            for data in train_loader:

                self.optimizer.zero_grad()
                data.to(self.device)

                # perform negative sampling per user so that we can use BPR loss
                pos_edge_label_index = data[
                    self.user_attr, self.action_attr, self.item_attr
                ].edge_label_index

                # to handle higher neg ratios, we will sample neg_ratio
                poss_items = data[self.item_attr].n_id

                neg_idx = torch.randint(
                    len(poss_items),
                    (pos_edge_label_index.shape[1] * self.neg_ratio,),
                    device=self.device,
                )
                neg_edge_label_index = torch.stack(
                    [pos_edge_label_index[0, :].repeat(self.neg_ratio), neg_idx], dim=0
                )

                # put positive (coming from dataloader) and neg samples together
                train_edge_label_index = torch.cat(
                    [
                        pos_edge_label_index,
                        neg_edge_label_index,
                    ],
                    dim=1,
                )
                # perform forward pass and get associated loss
                loss = self.forward_and_get_loss(data, train_edge_label_index, epoch=e)

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * train_edge_label_index.shape[1]
                total_examples += train_edge_label_index.shape[1]
            print(f"Phase {self.phase} Epoch: {e:03d}, Loss: {total_loss / total_examples:.4f}")

            train_loss.append(total_loss / total_examples)

            t_end_epoch = time.time()
            times.append(t_end_epoch - t_start_epoch)
    
            ### Validation step ###
            with torch.no_grad():
                val_data.to(self.device)

                # need to get the train data to zero out in the rec matrix
                train_edge_label_index = train_data[
                    self.user_attr, self.action_attr, self.item_attr
                ][f"link_index"]

                # need to get the val data for validation metric
                val_edge_label_index = val_data[
                    self.user_attr, self.action_attr, self.item_attr
                ][f"link_index"]

                # get ndgc metric on validation set
                ndcg = self.forward_and_get_ndgc(
                    val_data, train_edge_label_index, val_edge_label_index
                )

                val_ndcg.append(ndcg)

            print(f"Epoch: {e:03d}, Val NDCG: {ndcg:.4f}")
            if (e % save_epoch_freq) == 0:
                model_checkpoints[e] = copy.deepcopy(self.model)

            # found a new best model
            if ndcg > best_val_ndcg:
                best_model = copy.deepcopy(self.model)
                best_val_ndcg = ndcg
                last_epoch = e
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                # check if we havent updated in X epochs
                if self.phase == 'tune':
                    # if in fine_tuning phase, we can break 
                    if (early_stop_counter >= early_stop_cutoff):
                        break 
                elif self.phase == 'warm':
                    # if using this to dictate warm start, will switch over to fine_tune
                    if self.warm_start_metric == 'val_loss':
                        if (early_stop_counter >= early_stop_cutoff): 
                            self.phase = 'tune'
                            early_stop_counter = 0
                            switch_over_epoch = e
                    # otherwise, something else will cause the flip 
        

        return (
            train_loss,
            val_ndcg,
            best_model,
            best_val_ndcg,
            last_epoch,
            switch_over_epoch,
            model_checkpoints,
            times
        )


def main():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument("--dataset_save_path", type=str, default="./datasets")
    parser.add_argument("--cache", type=str2bool, default=True)
    # model parameters
    parser.add_argument(
        "--model", type=str, default="LGConv", choices=["LGConv", "MLP"]
    )
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--model_save_path", type=str, default="./models_chkp_warmstart"
    )

    # training parameters
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--loss", type=str, default="BPR", choices=["BPR", "SSM", "align"])
    parser.add_argument(
        "--reg_types",
        type=str,
        default=None,
        help="pass delimited string of regularization terms to be parsed",
    )
    parser.add_argument(
        "--gamma_vals",
        type=str,
        default=None,
        help="pass delimited string of reg strength coeffs",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=True,
        help="Whether to overwrite previously trained models",
    )
    parser.add_argument(
        "--warm_start_reg_types",
        type=str,
        default=None,
        help="regularization to use during warm start, as opposed to full training",
    )
    parser.add_argument(
        "--warm_start_gamma_vals",
        type=str,
        default=None,
        help="coeffecients to use during warm start, as opposed to full training",
    )
    parser.add_argument(
        "--warm_start_epochs",
        type=int,
        default=100,
        help="Number of epochs to warm start",
    )
    parser.add_argument(
        "--warm_start_metric",
        type=str,
        default=None,
        help="If not none, this is the metric we will use to flip over to the fine-tuning phase of training"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="number of epochs to use before early stopping"
    )

    args = parser.parse_args()

    # parse and format regs and gammas
    args.regs = [r for r in args.reg_types.split(",") if r != "-1"]
    args.gammas = [float(g) for g in args.gamma_vals.split(",") if g != "-1"]

    if args.warm_start_reg_types:
        args.warm_start_regs = [
            r for r in args.warm_start_reg_types.split(",") if r != "-1"
        ]
    else:
        args.warm_start_regs = []
    if args.warm_start_gamma_vals:
        args.warm_start_gammas = [
            float(g) for g in args.warm_start_gamma_vals.split(",") if g != "-1"
        ]
    else:
        args.warm_start_gammas = []

    if (len(args.regs) != len(args.gammas)) or (
        len(args.warm_start_regs) != len(args.warm_start_gammas)
    ):
        print(
            "Regularization terms and weighting coefficients need to match in length."
        )
        sys.exit()

    # check if we have already processed this set of hyperparameters
    processed_set = get_processed_params(
        args.dataset,
        args.model,
        args.loss,
        args.reg_types.replace(",", "_") if args.reg_types != "-1" else "",
    )
    params_to_check = tuple(
        [
            args.seed,
            args.num_layers,
            args.hidden_dim,
            args.lr,
            args.weight_decay,
            args.neg_ratio,
            args.loss,
        ]
    )

    if (params_to_check in processed_set) and (args.overwrite is False):
        print("Already processed these hyperparameters")
        sys.exit()

    seed_everything(args.seed)
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
        "_" + args.reg_types.replace(",", "_") if args.reg_types != "-1" else ""
    )

    if not os.path.exists(args.base_model_save_path):
        os.makedirs(args.base_model_save_path)

    # dataset object holds the additional info to index into data object
    dataset = Data(
        args.dataset, args.dataset_save_path, split_save_folder
    )

    # gets train, val, and test data. Train will be in batches, rest will be as data objects
    train_loader, train_data, val_data = dataset.get_dataloaders(
        num_layers=args.num_layers if args.model != "MLP" else 0,
        num_neighs=-1,
        batch_size=args.batch_size,
        testing=False,
        seed=args.seed,
    )
    print(train_data)

    # need to expose some properties of the data to the model
    num_users = train_data[dataset.user].num_nodes
    num_items = train_data[dataset.item].num_nodes
    dataset.data = train_data

    # sets up the model for us, including the encoder and decoder steps
    model = RecModel(
        dataset,
        num_users,
        num_items,
        args.model,
        hidden_dim=args.hidden_dim,
        depth=args.num_layers,
        device=device,
    )

    args.epochs = int(args.epochs)
    # Set up trainer that will orchestrate training
    trainer = Trainer(args.model, model, device, dataset, args)

    # Train the model
    (
        train_loss,
        val_ndcg,
        best_model,
        best_val_ndcg,
        last_epoch,
        switch_over_epoch, 
        model_checkpoints,
        times
    ) = trainer.train_model(train_loader, train_data, val_data, epochs=args.epochs, early_stop_cutoff=args.patience)

    # save all metadata and results for these runs
    curr_time = datetime.now().strftime("%m-%d-%Y:%H:%M:%S:%f")
    args.model_save_path = args.base_model_save_path + "/" + curr_time + ".pt"
   
    for e in model_checkpoints:
        curr_model = model_checkpoints[e]
        torch.save(
            curr_model.state_dict(),
            args.base_model_save_path + "/" + curr_time + f"_epochs_{e}.pt",
        )
    torch.save(best_model.state_dict(), args.model_save_path)

    arg_dict = vars(args)
    args.model_param_save_path = args.base_model_save_path + "/" + curr_time + ".params"
    arg_dict["train_loss"] = train_loss
    arg_dict["val_ndcg"] = val_ndcg
    arg_dict["best_val_ndcg"] = best_val_ndcg
    arg_dict["last_epoch"] = last_epoch
    arg_dict["switch_over_epoch"] = switch_over_epoch 
    arg_dict["times"] = times 

    with open(args.model_param_save_path, "w") as f:
        json.dump(arg_dict, f)


if __name__ == "__main__":
    main()
