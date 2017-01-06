import numpy as np
from utilities import extract, batch_iter, generate_training_set
from sklearn import metrics, linear_model, model_selection, preprocessing, ensemble, svm, naive_bayes, neighbors
import sys
import pandas as pd

INPUT_PATH = 'input/'
MODEL_PATH = 'models/'
train, label, data = extract(INPUT_PATH + 'train.csv', target='label')

regressors = np.column_stack(
    (np.ones(shape=train.shape[0]), train.apply(preprocessing.scale, axis=0, with_mean=True, with_std=True))
)

regressand = np.array(data['label'])

lr = linear_model.LogisticRegression(fit_intercept=False)
# lr.fit(regressors, regressand)
# print(lr.coef_)

KF_generator = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

# avg_scores = model_selection.cross_val_score(
#     lr, regressors, regressand, scoring='accuracy', cv=KF_generator)

# print('Using given features by Kaggle, Logistic Regression model accuracy is: ', end='')
# print('{1} averaging in {0:.2f}%'.format(100 * np.mean(avg_scores), avg_scores), flush=True, end='\n')

regressors = train.apply(preprocessing.scale, axis=0, with_mean=True, with_std=True)

forest = ensemble.RandomForestClassifier(n_estimators=300, max_depth=1000, max_leaf_nodes=1000)

avg_scores = model_selection.cross_val_score(
    forest, regressors, regressand, scoring='accuracy', cv=KF_generator)

print('Using given features by Kaggle, Random Forest model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(avg_scores), avg_scores), flush=True, end='\n')


regressors = train.apply(preprocessing.scale, axis=0, with_mean=True, with_std=True)

bayes = naive_bayes.BernoulliNB()

avg_scores = model_selection.cross_val_score(
    bayes, regressors, regressand, scoring='accuracy', cv=KF_generator)

print('Using given features by Kaggle, Naive Bayes model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(avg_scores), avg_scores), flush=True, end='\n')


clf = svm.SVC()
avg_scores = model_selection.cross_val_score(
    clf, regressors, regressand, scoring='accuracy', cv=KF_generator)

print('Using given features by Kaggle, SVM model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(avg_scores), avg_scores), flush=True, end='\n')

neigh = neighbors.KNeighborsClassifier(n_neighbors=3)

avg_scores = model_selection.cross_val_score(
    neigh, regressors, regressand, scoring='accuracy', cv=KF_generator)

print('Using given features by Kaggle, K-nearest Neighbour model accuracy is: ', end='')
print('{1} averaging in {0:.2f}%'.format(100 * np.mean(avg_scores), avg_scores), flush=True, end='\n')
