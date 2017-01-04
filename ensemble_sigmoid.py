import numpy as np
from utilities import extract, batch_iter, generate_training_set
from sklearn import metrics, linear_model, model_selection, preprocessing
import sys
import pandas as pd


INPUT_PATH = 'input/'
MODEL_PATH = 'models/'
train, label, data = extract(INPUT_PATH + 'train.csv', target='label')


regressors = np.column_stack(
    (np.ones(shape=train.shape[0]), train.apply(preprocessing.scale, axis=0, with_mean=False, with_std=True))
)

regressand = np.array(data['label'])

lr = linear_model.LogisticRegression(fit_intercept=False)
lr.fit(regressors, regressand)
print(lr.coef_)

KF_generator = model_selection.KFold(n_splits=5, shuffle=False)
avg_scores = model_selection.cross_val_score(
    lr, regressors, regressand, scoring='accuracy', cv=KF_generator)

print('Using given features by Kaggle, Logistic Regression model accuracy is: ', end='')
print('{0:.2f}%'.format(100 * np.mean(avg_scores)), flush=True, end='\n')
