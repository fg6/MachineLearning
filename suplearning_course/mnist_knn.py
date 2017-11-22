#import numpy as np	
from datetime import datetime
#import pandas as pd
import matplotlib.pyplot as plt

from read import get_data
from knn import KNN


T0 = datetime.now()

train_size=1000
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
yw=[]
Twy=[]
for k in range(1,2):
    
    knn = KNN(k)
    knn.fit(train_ims, train_target)

    t0  = datetime.now()
    print 'k=', k, ' Train results:'
    score=knn.score(train_ims, train_target)
    print ' k =', k, 'score =', score
    print " Train time:", (datetime.now() - t0)
    Ty.append(score)
    
    t0  = datetime.now()
    print 'k=', k, ' Test results:'   
    score=knn.score(test_ims, test_target)
    print ' k =', k, 'score =', score
    print " Test time:", (datetime.now() - t0)

    x.append(k)
    y.append(score)


    t0  = datetime.now()
    print 'k=', k, ' Train results:'
    wscore=knn.wscore(train_ims, train_target)
    print ' k =', k, 'score =', wscore
    print " Train time:", (datetime.now() - t0)
    Twy.append(wscore)
    
    t0  = datetime.now()
    print 'k=', k, ' Test results:'   
    score=knn.wscore(test_ims, test_target)
    print ' k =', k, 'score =', wscore
    print " Test time:", (datetime.now() - t0)
    yw.append(score)
    

test=plt.scatter(x,y,color='g', label='Test Set')
train=plt.scatter(x,Ty,color='b', label='Train Set')
wtest=plt.scatter(x,yw,color='r', label='Test Set W')
wtrain=plt.scatter(x,Twy,color='salmon', label='Train Set W')
plt.legend(loc='lower left')

plt.show()
        



