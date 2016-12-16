import pandas as pd
import numpy as np


def extract(data):
    data = pd.read_csv(data)
    label = pd.get_dummies(data['label'])
    train = data.drop(['label'], axis=1)
    return train, label, data


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """batch iterator"""

    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            new_data = np.random.permutation(data)
        else:
            new_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min(start_index + batch_size, data_size)
            if start_index == end_index:
                break
            else:
                yield epoch, batch_num, new_data[start_index:end_index]

