import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups_vectorized
import sys
import torch.nn as nn
import torch.nn.functional as F


class UpperModel(torch.nn.Module):

    def __init__(self, n_feats, device):
        super(UpperModel, self).__init__()
        self.x = torch.nn.Parameter(
            torch.zeros(n_feats, requires_grad=True, device=device).requires_grad_(True)
        )

    def forward(self):
        return self.x


class LowerModel(torch.nn.Module):

    def __init__(self, n_feats, device, num_classes):
        super(LowerModel, self).__init__()
        self.y = torch.nn.Parameter(
            torch.zeros((n_feats, num_classes), requires_grad=True, device=device)
        )
        self.y.data = nn.init.kaiming_normal_(self.y.data.t(), mode="fan_out").t()

    def forward(self):
        return self.y


def evaluate(x, w, testset):
    with torch.no_grad():
        test_x, test_y = testset
        y = test_x.mm(x)
        loss = F.cross_entropy(y, test_y).detach().item()
        acc = (y.argmax(-1).eq(test_y).sum() / test_y.shape[0]).detach().cpu().item()
    return loss, acc


def get_data(args):
    def from_sparse(x):
        x = x.tocoo()
        values = x.data
        indices = np.vstack((x.row, x.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = x.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))

    val_size = 0.5
    train_x, train_y = fetch_20newsgroups_vectorized(
        subset="train",
        return_X_y=True,
        data_home=args.data_path,
        download_if_missing=True,
    )

    test_x, test_y = fetch_20newsgroups_vectorized(
        subset="test",
        return_X_y=True,
        data_home=args.data_path,
        download_if_missing=True,
    )

    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, stratify=train_y, test_size=val_size
    )
    test_x, teval_x, test_y, teval_y = train_test_split(
        test_x, test_y, stratify=test_y, test_size=0.5
    )

    train_x, val_x, test_x, teval_x = map(
        from_sparse, [train_x, val_x, test_x, teval_x]
    )
    train_y, val_y, test_y, teval_y = map(
        torch.LongTensor, [train_y, val_y, test_y, teval_y]
    )

    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)
