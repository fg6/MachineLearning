#import numpy as np	
from datetime import datetime
#import pandas as pd
import matplotlib.pyplot as plt

from read import get_data
#from naivebayes import NaiveBayes
import naivebayes
reload(naivebayes)

plot=0

T0 = datetime.now()

try: 
    train_size
except NameError:
    train_size=10000
    test_size=1000
    size=train_size+test_size
    ims,target=get_data(size)

    train_ims=ims[0:train_size,]
    train_target=target[0:train_size,]
    test_ims=ims[train_size:size,]
    test_target=target[train_size:size,]




x=[]
y=[]
Ty=[]
    
model = naivebayes.NaiveBayes()
model.fit(train_ims, train_target)

score=model.score(train_ims, train_target)
print("train:",score)
score=model.score(test_ims, test_target)
print("test",score)

if plot:
    test=plt.scatter(x,y,color='g', label='Test Set')
    train=plt.scatter(x,Ty,color='b', label='Train Set')
    plt.legend(loc='lower left')
    plt.show()
        



