import numpy as np

Why = np.ones([81,2])*0.1
do = np.ones([81,1])*0.012345679012345678
do[0] = -0.9876543209876543
print(Why)
print(do)

dh = np.dot(Why.T,do)
print (dh)