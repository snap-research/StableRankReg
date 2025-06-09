import math

import numpy as np
import torch

def NDCG(user, num_items):
    if sum(user) == 0:
        # no relevant items in the rec, just return 0
        return 0

    # get ideal NCG based on number of items for user
    ideal_user = np.zeros(user.shape)
    ideal_user[:num_items] = 1

    idcg = 0
    for i in range(1, len(ideal_user) + 1):
        idcg = idcg + (ideal_user[i - 1] / math.log2(i + 1))

    dcg = 0
    # discounted gain will logarithmically decay score for each rec
    for i in range(1, len(user) + 1):
        dcg = dcg + (user[i - 1] / math.log2(i + 1))

    ndcg = dcg / idcg
    return ndcg


def compute_ndcg_at_k(top_k_recs, edge_index):

    ndcg_user = []

    num_edges = edge_index.shape[1]
    start = 0
    end = 1

    binary_matrix = []
    pos_len = []
    while end < num_edges:
        # while we are on the same source node, move end index up one
        if edge_index[0, start] == edge_index[0, end]:
            end += 1
        else:
            # when they are not the same, we will compute NDCG from start to end and move start up
            user = edge_index[0, start]
            items = edge_index[1, start:end].cpu().detach().numpy()

            recs = top_k_recs[user, :].cpu().detach().numpy()

            # convert recs into binary ranking list based on items
            mask = np.isin(recs, items).astype(int)
            ndcg = NDCG(mask, min(top_k_recs.shape[1], len(items)))

            ndcg_user.append(ndcg)
            start = end
            end += 1

    avg_ndcg = sum(ndcg_user) / len(ndcg_user)

    return (
        avg_ndcg,
        np.array(ndcg_user),
    )


def compute_recall_at_k(rec_matrix, edge_index, k=20, per_user=True):
    # recall at k on recommendations

    correct = total = 0

    if per_user:
        recall_at_k_scores = []
        degrees = []

    if rec_matrix.shape[1] != k:
        top_k_recs = torch.topk(rec_matrix, k=k, dim=1).indices
    else:
        top_k_recs = rec_matrix
    num_edges = edge_index.shape[1]
    start = 0
    end = 1
    users = []

    while end < num_edges:
        # while we are on the same source node, move end index up one
        if edge_index[0, start] == edge_index[0, end]:
            end += 1
        else:
            # when they are not the same, we will compute the recall from start to end and move start up
            user = edge_index[0, start]
            items = edge_index[1, start:end]

            recs = top_k_recs[user, :]

            # compute the relevant items
            relevant_items = np.intersect1d(
                recs.cpu().detach().numpy(), items.cpu().detach().numpy()
            )

            interact_items_at_k = len(items)

            correct += len(relevant_items)
            total += min(k, interact_items_at_k)

            if per_user:
                recall_at_k_scores.append(
                    float(len(relevant_items)) / interact_items_at_k
                )
                degrees.append(len(items))

            start = end
            end += 1
            users.append(user.item())

    recall_at_k_score = float(correct) / total

    return (
        recall_at_k_score,
        np.array(recall_at_k_scores) if per_user else None,
        np.array(users) if per_user else None,
        np.array(degrees) if per_user else None,
    )
