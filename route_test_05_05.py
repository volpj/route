import time

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import math


def B_Curve(M, Li):
   p  = M 
   n  = M.shape[0]
   n1 = n - 1   
   sigma = np.zeros(n)
   ind = np.arange(n)  
   for i in ind:
       sigma[i] = math.factorial(n1) / (math.factorial(i)*math.factorial(n1-i))
   maat = Li.shape
   print(maat[0])
   s=(maat[0]+1,4)
   l=np.zeros(s)
   s=(1,n)
   UB = np.zeros(s)
   print(l.shape,' ',UB.shape)
   i = 0
   for u in np.arange(0., 1+eps, 1/maat[0]): #eps is niet duidelijk, door proberen
       for d in ind:  #d gaat van 0->n1 ipv 1->n =>
           UB[0,d] = sigma[d] * (1.-u)**(n1-d) * u**d
       l[i,] = UB[0,]
       i = i + 1
   return np.dot(l,p)



plot = False

start = time.time()

eps = 2.2204e-16
bk  = 200
cpd = 300
avd = 400
d2r = 0.01745329
T = [-1000, 1000, 0, 0, 0, 0]

Txy = np.array([[T[0]],[T[1]]])


#denk, of
K = np.array([[0],[0]])
# K.shape = 2,1 -> K[0,0] en K[1,0]

#of net als Txy
#K = np.array([0,0])
# K.shape = 2, -> K[0,] en K[1,]

data = pd.read_excel(r'data/Lidar_excel_4.xlsx')
df = pd.DataFrame(data)

Lip = df["Correction_Angle"]*d2r
Lir = df["Distance"]

Lix = Lir*np.cos(Lip)
Liy = Lir*np.sin(Lip)

Li0 = pd.DataFrame({'x': Lix, 'y': Liy})
Li  = Li0[~(abs(Li0['y']) <= eps)]  

cluster = DBSCAN(eps=bk, min_samples=5).fit(Li)
labels = cluster.labels_ + 1

XYi = pd.DataFrame({'x': Li.x, 'y':Li.y, 'label': labels})
groups = XYi.groupby('label')

# Plot
if plot:
   fig, ax = plt.subplots()
   for name, group in groups:
      ax.plot(group.x, group.y, marker='o', linestyle='', ms=2, label=name)
   ax.legend()
   plt.show()
#

M = np.array([ [K[0,0]  , K[1,0]],
               [K[0,0]  , K[1,0]+cpd],
               [Txy[0,0], Txy[1,0]-cpd],
               [T[0]    , T[1]] ])

#3e punt is overlappend aan 2e om matrixmaat constant te houden.
M[2,0] = cpd*np.sin(T[5]) + T[0];
M[2,1] =-cpd*np.cos(T[5]) + T[1];

P = B_Curve(M, Li)




#hold on;
#axis([-2000 2000 -2000 2000]);
#grid on;


end = time.time()
print(end - start)








