import numpy as np	
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans

def most_common(lst):
    return max(set(lst), key=lst.count)

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

def assign(target, clusters):
    
    df = pd.DataFrame({'targets' : target, 'clusters' : clusters, })
    df.sort_values('clusters')

    clus_assign=dict()
    clus_list=dict()

    for index, row in df.iterrows():
        targ=row['targets']
        clus=row['clusters']
    
        if clus in clus_list.keys():
            clus_list[clus].append(targ)
        else:
            temp = list()
            temp.append(targ)
            clus_list[clus] = temp  

    for key,value in clus_list.items():
        clus_assign[key] = most_common(value)

    return clus_assign

try: 
    size
except NameError:
    ims, target = get_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_train.csv')
    kaggle_test = kaggle_data('/Users/frangy/Documents/DataAna/kaggle_data/mnist_test.csv')
    size=len(target)

resize=0

if resize:
    train_size=10000
    test_size=1000
    
    train_ims=ims[0:train_size,]
    train_target=target[0:train_size,]
    test_ims=ims[train_size:size,]
    test_target=target[train_size:size,]

dofit=1

if dofit:
    nclus=set(train_target)

    np.random.seed(42)
    #model = KMeans(n_clusters=len(nclus), init='random', n_init=10)
    model = KMeans(n_clusters=len(nclus), init='k-means++', n_init=10)
    model.fit(train_ims)
    clusters = model.fit_predict(train_ims)

# assign a target to a cluster
clus_assign = assign(train_target, clusters)

# train
train_clusters = model.predict(train_ims)
results = [ clus_assign[k] for k in train_clusters ]
score = np.mean(results == train_target)
print("train:",score)

test_clusters = model.predict(test_ims)
results = [ clus_assign[k] for k in test_clusters ]

score = np.mean(results == test_target)
print("test:",score)

