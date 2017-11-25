import numpy as np	
from datetime import datetime
import pandas as pd
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.grid_search import GridSearchCV


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
    kaggle_test = kaggle_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_test.csv')
    size=len(target)

resize=1
if resize:
    train_size=30000
    test_size=10000
    
    train_ims=ims[0:train_size,]
    train_target=target[0:train_size,]
    test_ims=ims[train_size:size,]
    test_target=target[train_size:size,]
    

def find_best():

    clf =  RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

    # Parameters values to test:
    param_grid = {        
        'n_estimators': [200, 700, 1000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    # Define test and fit!
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
    CV_rfc.fit(train_ims, train_target)

    # Get best parameters:
    print '\n',CV_rfc.best_params_   
    return CV_rfc.best_estimator_.n_estimators, CV_rfc.best_estimator_.max_features, CV_rfc.best_estimator_.criterion


# find best parameters from a list of possibilities:
n_est, maxf, crit, min_s_leaf = find_best()


n_est = 1000  
maxf = 'log2'
crit = 'gini'
min_s_leaf = 1  


# Use the best parameters found to build a model and fit train data:
model =  RandomForestClassifier(n_jobs=-1, max_features= maxf, n_estimators=n_est, criterion = crit, min_samples_leaf = min_s_leaf, oob_score = True) 
model.fit(train_ims, train_target)
    

# Estimate model accuracy for train data:
results = model.predict(train_ims)
score = metrics.accuracy_score(train_target,results)
print("Accuracy on train data:",score)

# Estimate model accuracy for test data:
results = model.predict(test_ims)
score =  metrics.accuracy_score(test_target,results)
print("Accuracy on test data:",score)


# Use current model to get prediction for the Kaggle test sample:
kaggle_predict=1
if kaggle_predict:

    results = model.predict(kaggle_test)
    f = open('randforest_prediction.txt', 'w')
    f.write('ImageId,Label\n')
    for i, r  in enumerate(results):
        s = str(i+1)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()
