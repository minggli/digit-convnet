from PIL import Image, ImageChops, ImageOps
import os
import shutil
import pandas as pd
import numpy as np


def extract(data):
    train = pd.read_csv(data)
    label = pd.get_dummies(train['label'])
    train.drop(['label'], axis=1, inplace=True)
    return train, label


def delete_folders(dirs=['test', 'train', 'validation'], dir_path='leaf/images/'):

    for directory in dirs:
        if os.path.exists(dir_path + directory):
            shutil.rmtree(dir_path + directory)


def pic_resize(f_in, size=(96, 96), pad=True):

    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)

    image_size = image.size

    if pad:
        thumb = image.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - image_size[0]) // 2, 0)
        offset_y = max((size[1] - image_size[1]) // 2, 0)

        thumb = ImageChops.offset(thumb, offset_x, offset_y)

    else:
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, (0.5, 0.5))

    return thumb


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


def move_classified(test_order, ans, mapping, dir_path='leaf/images/'):
    answers = dict()
    for k, i in enumerate(test_order):
        quest = list(ans[k]).index(max(list(ans[k])))
        name = mapping[quest]
        answers[i] = name

    for filename in answers.keys():
        pid = int(filename.split('.')[0])
        directory = dir_path + 'test/' + answers[filename]
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.rename(dir_path + 'test/' + filename, directory + '/' + filename)

