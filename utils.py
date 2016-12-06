import pandas as pd
import numpy as np


def extract(data):
    train = pd.read_csv(data)
    label = pd.get_dummies(train['label'])
    train.drop(['label'], axis=1, inplace=True)
    return train, label


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """batch iterator"""
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            new_data = np.random.permutation(data)
        else:
            new_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield epoch, batch_num, new_data[start_index:end_index]

