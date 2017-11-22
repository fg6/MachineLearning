import matplotlib.pyplot as plt

from read import get_xor
from knn import KNN


ims,target=get_xor()

plt.scatter(ims[:,0],ims[:,1],s=100, c=target)
plt.show()

for k in range(1,5):   
    model = KNN(k)
    model.fit(ims, target)
    score = model.wscore(ims, target)
    print ' k =', k, 'score =', score

  
