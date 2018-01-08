import numpy as np	
from datetime import datetime
import pandas as pd

import sys
sys.path.append('../../mypackages/')
import utils

def mnist_data(file, train):
    df = pd.read_csv(file)
    data = df.as_matrix()
    np.random.shuffle(data)
 
    from sklearn import preprocessing

    if train:
        X = data[:, 1:] / 255.0 # data is from 0..255
        normalized_X = preprocessing.normalize(X, norm='l2')
        standardized_X = preprocessing.scale(X)
        Y = data[:, 0]
        #return standardized_X, Y
        return X, Y
    else:
        X = data[:, :] / 255.0 # data is from 0..255
        return X


X, y = mnist_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_train.csv',1)
kaggle_test = mnist_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_test.csv',0)
size=len(y)

    
test_perc=0.5  # perc size of test sample
X_train, X_test, y_train, y_test = utils.resize(X, y, test_perc)

kaggle_predict=0
findbest=0
 
if findbest:
    n_est, maxf, crit, min_s_leaf = utils.best_randforest_cl(X, y)
else:
    n_est = 1000  
    maxf = 'log2'
    crit = 'gini'
    min_s_leaf = 1  

clf_rf = utils.randforest_cl(X_train, y_train, n_est, maxf, crit, min_s_leaf)

y_pred = clf_rf.predict(X_train)
print('Training score:')
utils.get_accuracy_cl(y_pred, y_train)
    
y_pred = clf_rf.predict(X_test)
print('Testing score:')
utils.get_accuracy_cl(y_pred, y_test)


n_est = 1000  
maxf = 'log2'
crit = 'gini'
min_s_leaf = 1  




if kaggle_predict:

    results = clf_rf.predict(kaggle_test)

    f = open('randforest_prediction2.txt', 'w')
    f.write('PassengerId,Survived\n')
    for i, r  in enumerate(results):
        k=kid[i]
        s = str(k)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()
