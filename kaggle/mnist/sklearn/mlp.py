import numpy as np	
from datetime import datetime
import pandas as pd

import sys
sys.path.append('../../mypackages/')
import utils

def mnist_data(file, train):
    df = pd.read_csv(file)
    data = df.as_matrix()
    
    if train:
        np.random.shuffle(data)
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

    
test_perc=0.5  # perc size of test sample
X_train, X_test, y_train, y_test = utils.resize(X, y, test_perc)

kaggle_predict=1
findbest=0

if findbest:
    hidden_layer_sizes,activation,solver,learning_rat = utils.best_mlp_cl(X, y)
    #The best parameters are {'activation': 'tanh', 'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (200,)} with a score of 0.97

else:
   hidden_layer_sizes = (800, )
   activation = "tanh"
   solver = "adam"
   learning_rate = "constant"

   
clf_rf = utils.mlp_cl(X_train, y_train, hidden_layer_sizes, activation, solver, learning_rate) 
y_pred = clf_rf.predict(X_train)
print(y_pred)
print('Training score:')
utils.get_accuracy_cl(y_pred, y_train)
    
y_pred = clf_rf.predict(X_test)
print('Testing score:')
utils.get_accuracy_cl(y_pred, y_test)

#Training score:
#('  Score :', 1.0, '\n  Pos_ok:', 937, 'False Neg:', 0, ' Pos error:', 0.0, '%\n  Neg_ok:', 829, 'False_pos:', 0, ' Neg error:', 0.0, '%')
#Testing score:
#('  Score :', 0.9480654761904762, '\n  Pos_ok:', 3649, 'False Neg:', 0, ' Pos error:', 0.0, '%\n  Neg_ok:', 3209, 'False_pos:', 0, ' Neg error:', 0.0, '%')

# Use current model to get prediction for the Kaggle test sample:
if kaggle_predict:
    results = clf_rf.predict(kaggle_test)
    f = open('mlp_prediction.txt', 'w')
    f.write('ImageId,Label\n')
    for i, r  in enumerate(results):
        s = str(i+1)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()
