#import numpy as np	
from datetime import datetime
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from read import get_data
#from naivebayes import NaiveBayes
import decisionTree
reload(decisionTree)

plot=0
fit=0

T0 = datetime.now()

try: 
    ims
except NameError:
 
    ims,target=get_data()
    
train_size=len(target)/2
test_size =  train_size
size=train_size+test_size
    
train_ims=ims[0:train_size,]
train_target=target[0:train_size,]
test_ims=ims[train_size:size,]
test_target=target[train_size:size,]
    
idx= np.logical_or(train_target == 0 ,train_target == 1)
binary_X = train_ims[idx]
binary_Y = train_target[idx]

idxtest= np.logical_or(test_target == 0 ,test_target == 1)
binary_Xtest = test_ims[idxtest]
binary_Ytest = test_target[idxtest]
    
x=[]
y=[]
Ty=[]
    
#model = decisionTree.TreeNode(7, 1)
model = decisionTree.DecisionTree(7)
model.fit(binary_X,binary_Y)


###model.fit(train_ims, train_target)
###model.score(train_ims, train_target)



print(len(binary_Y))
score=model.score(binary_X,binary_Y)
print("Binary, train score:", score)
#print(len(binary_Ytest))
score_test=model.score(binary_Xtest,binary_Ytest)
print("Binary, test score:", score_test)

if fit:
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
        



