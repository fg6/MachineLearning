import numpy as np	
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

#from read import get_data
import naivebayes
reload(naivebayes)

plot=0

T0 = datetime.now()

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
    train_size=20000
    test_size=20000
    
    train_ims=ims[0:train_size,]
    train_target=target[0:train_size,]
    test_ims=ims[train_size:size,]
    test_target=target[train_size:size,]

    kaggle_test = kaggle_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_test.csv')

    
model = naivebayes.Bayes()
model.fit(train_ims, train_target)

score=model.score(train_ims, train_target)
print("train:",score)
score=model.score(test_ims, test_target)
print("test",score)

kaggle_predict=0

if kaggle_predict:

    results = model.predict(kaggle_test)

    f = open('bayes_prediction.txt', 'w')
    f.write('ImageId,Label\n')
    for i, r  in enumerate(results):
        s = str(i+1)+str(',')+str(r)+str('\n')
        f.write(s)
    f.close()

