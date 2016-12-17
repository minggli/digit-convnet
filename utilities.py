import pandas as pd
import numpy as np


def extract(data, target=None):
    data = pd.read_csv(data)
    if target in data.columns:
        label = pd.get_dummies(data[target])
        train = data.drop([target], axis=1)
    else:
        label = None
        train = None
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


def generate_training_set(data, label, test_size=0.05):

    index = len(data)
    random_index = np.random.permutation(index)

    train_size = int((1 - test_size) * index)

    train_index = random_index[:train_size]
    test_index = random_index[train_size:]

    x_train = np.array(data.ix[train_index, :])
    y_train = np.array(label.ix[train_index, :])

    x_valid = np.array(data.ix[test_index, :])
    y_valid = np.array(label.ix[test_index, :])

    combined_train = np.array([(x_train[i], y_train[i]) for i in range(len(train_index))])
    combined_valid = np.array([(x_valid[i], y_valid[i]) for i in range(len(test_index))])

    return combined_train, combined_valid
