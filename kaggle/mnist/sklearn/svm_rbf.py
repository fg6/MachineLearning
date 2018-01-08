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
 

    if train:
        X = data[:, 1:] / 255.0 # data is from 0..255
        Y = data[:, 0]
        return X, Y
    else:
        X = data[:, :] / 255.0 # data is from 0..255
        return X


try: 
    size
except NameError:
    X, y = mnist_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_train.csv',1)
    kaggle_test = mnist_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_test.csv',0)
    size=len(y)

    
test_perc=0.8  # perc size of test sample
X_train, X_test, y_train, y_test = utils.resize(X, y, test_perc)

kaggle_predict=0
findbest=0

#if findbest:
#    hidden_layer_sizes,activation,solver,learning_rat = utils.best_mlp_cl(X, y)
#else:
#   hidden_layer_sizes = (100, )
#   activation = "relu"
#   solver = "sgd"
#   learning_rate ="constant"

   
clf_rf = utils.svm_rbf(X_train, y_train)
y_pred = clf_rf.predict(X_train)
print(y_pred)
print('Training score:')
utils.get_accuracy_cl(y_pred, y_train)
    
y_pred = clf_rf.predict(X_test)
print('Testing score:')
utils.get_accuracy_cl(y_pred, y_test)

#Training score:
#('  Score :', 0.934047619047619, '\n  Pos_ok:', 2348, 'False Neg:', 0, ' Pos error:', 0.0, '%\n  Neg_ok:', 1966, 'False_pos:', 0, ' Neg error:', 0.0, '%')
#Testing score:
#('  Score :', 0.9287142857142857, '\n  Pos_ok:', 2248, 'False Neg:', 1, ' Pos error:', 0.0, '%\n  Neg_ok:', 2067, 'False_pos:', 0, ' Neg error:', 0.0, '%')


#SLOW!
# Use current model to get prediction for the Kaggle test sample:
#if kaggle_predict:
#   results = model.predict(kaggle_test)
#    f = open('mlp_prediction.txt', 'w')
#    f.write('ImageId,Label\n')
#    for i, r  in enumerate(results):
#        s = str(i+1)+str(',')+str(r)+str('\n')
#        f.write(s)
#    f.close()
