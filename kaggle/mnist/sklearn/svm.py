import numpy as np	
from datetime import datetime
import pandas as pd

from sklearn import svm

def get_data(file):
    df = pd.read_csv(file)
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    return X, Y

def kaggle_data(file):
    df = pd.read_csv(file)
    data = df.as_matrix()
    X = data[:, :] / 255.0 # data is from 0..255
    return X


try: 
    size
except NameError:
    ims, target = get_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_train.csv')

    size=len(target)
    train_size=10000
    test_size=10000
    
    train_ims=ims[0:train_size,]
    train_target=target[0:train_size,]
    test_ims=ims[train_size:size,]
    test_target=target[train_size:size,]

    kaggle_test = kaggle_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_test.csv')

find_best=0

if find_best:
    # Import GridSearchCV
    from sklearn.grid_search import GridSearchCV

    # Set the parameter candidates
    parameter_candidates = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    
    # Create a classifier with the parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    
    # Train the classifier on training data
    clf.fit(train_ims, train_target)

    # Print out the results 
    print('Best score for training data:', clf.best_score_)
    print('Best `C`:',clf.best_estimator_.C)
    print('Best kernel:',clf.best_estimator_.kernel)
    print('Best `gamma`:',clf.best_estimator_.gamma)
    # Apply the classifier to the test data, and view the accuracy score
    clf.score(test_ims, test_target)



# Train and score a new classifier with the grid search parameters
#svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(train_ims, train_target).score(test_ims, test_target)


if 1:

    model = svm.SVC(gamma=0.001, C=10., kernel='rbf')
    model.fit(train_ims, train_target)

    results = model.predict(train_ims)
    score1 = model.score(train_ims,train_target)
    score = np.mean(results == train_target)
    print("train:",score,score1)

    results = model.predict(test_ims)
    score1 = model.score(test_ims, test_target)
    score = np.mean(results == test_target)
    print("test",score,score1)

kaggle_predict=0

if kaggle_predict:

    results = model.predict(kaggle_test)

    f = open('bayes_prediction.txt', 'w')
    f.write('ImageId,Label\n')
    for i, r  in enumerate(results):
        s = str(i+1)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()

