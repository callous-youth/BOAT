import copy
import random

import numpy as np
import torch


class Dataset:
    def __init__(self, data, target, polluted=False, rho=0.0):
        self.data = data.float() / torch.max(data)
        print(list(target.shape))
        if not polluted:
            self.clean_target = target
            self.dirty_target = None
            self.clean = np.ones(list(target.shape)[0])
        else:
            self.clean_target = None
            self.dirty_target = target
            self.clean = np.zeros(list(target.shape)[0])
        self.polluted = polluted
        self.rho = rho
        self.set = set(target.numpy().tolist())

    def data_polluting(self, rho):
        assert self.polluted == False and self.dirty_target is None
        number = self.data.shape[0]
        number_list = list(range(number))
        random.shuffle(number_list)
        self.dirty_target = copy.deepcopy(self.clean_target)
        for i in number_list[:int(rho * number)]:
            dirty_set = copy.deepcopy(self.set)
            dirty_set.remove(int(self.clean_target[i]))
            self.dirty_target[i] = random.randint(0, len(dirty_set))
            self.clean[i] = 0
        self.polluted = True
        self.rho = rho

    def data_flatten(self):
        try:
            self.data = self.data.view(self.data.shape[0], self.data.shape[1] * self.data.shape[2])
        except BaseException:
            self.data = self.data.reshape(self.data.shape[0],
                                          self.data.shape[1] * self.data.shape[2] * self.data.shape[3])

    # def get_batch(self,batch_size):

    def to_cuda(self):
        self.data = self.data.cuda()
        if self.clean_target is not None:
            self.clean_target = self.clean_target.cuda()
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.cuda()


def data_splitting(dataset, tr, val, test):
    assert tr + val + test <= 1.0 or tr > 1

    number = dataset.targets.shape[0]
    number_list = list(range(number))
    random.shuffle(number_list)
    if tr < 1:
        tr_number = tr * number
        val_number = val * number
        test_number = test * number
    else:
        tr_number = tr
        val_number = val
        test_number = test

    train_data = Dataset(dataset.data[number_list[:int(tr_number)], :, :],
                         dataset.targets[number_list[:int(tr_number)]])
    val_data = Dataset(dataset.data[number_list[int(tr_number):int(tr_number + val_number)], :, :],
                       dataset.targets[number_list[int(tr_number):int(tr_number + val_number)]])
    test_data = Dataset(
        dataset.data[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)], :, :],
        dataset.targets[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)]])
    return train_data, val_data, test_data
