import os

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

def generate_pyg_data(dataset):

    num_users = 0
    num_items = 0

    edge_index = []
    with open(f"{dataset}/train.txt") as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip("\n").split(" ")
                items = [int(i) for i in l[1:]]
                uid = int(l[0])

                num_items = max(num_items, max(items))
                num_users = max(num_users, uid)

                for item in items:
                    edge_index.append([uid, item])

    with open(f"{dataset}/test.txt") as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip("\n").split(" ")
                items = [int(i) for i in l[1:]]
                uid = int(l[0])

                for item in items:
                    edge_index.append([uid, item])

    edge_index = np.array(edge_index).T

    data = HeteroData()
    data["user"].num_nodes = num_users + 1
    data["item"].num_nodes = num_items + 1

    # re-index so that both user and items start at 0
    data["user", "rates", "item"].edge_index = torch.Tensor(edge_index).long()

    transform = T.ToUndirected()
    # create the new inverse and rename the inverse action for the dataloader to use
    data = transform(data)
    torch.save(data, f"{dataset}/data.pt")


if __name__ == "__main__":
    # this will be used to create datasets for Yelp2018 and Gowalla

    datasets = ["Gowalla", "Yelp2018"]

    for dataset in datasets:
        generate_pyg_data(dataset)
