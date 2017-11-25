import numpy as np	
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split

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
    train_size=10000
    test_size=10000
    
    train_ims=ims[0:train_size,]
    train_target=target[0:train_size,]
    test_ims=ims[train_size:size,]
    test_target=target[train_size:size,]

dofit=1

if dofit:
    model = LogisticRegression(C=1000.0, random_state=0)
    model.fit(train_ims, train_target)
    

predict=1
if predict:
    results = model.predict(train_ims)
    score1 = metrics.accuracy_score(train_target,results)
    score = np.mean(results == train_target)
    print("train:",score,score1)

    results = model.predict(test_ims)
    score1 =  metrics.accuracy_score(test_target,results)
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
