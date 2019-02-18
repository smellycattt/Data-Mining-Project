import csv
import math
import copy
import glob
import numpy as np
#   import pandas as pd
#from sklearn.preprocessing import scale
#from sklearn.metrics.pairwise import euclidean_distances
import sys
import pylab as plt
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()

#allfiles=glob.glob(imputpath+"*.csv")

max_Iter = 10
clusters=[]
clusters1=[]
def dist(cen,l):
  dis=9999999
  disf=0
  for i in range(0,len(l)):
    for j in range(0,len(cen)):
        dis1=pow(pow((cen[j][0]-l[j][0]),2)+pow((cen[j][1]-l[j][1]),2),1/2)
        if dis>dis1:
          dis=dis1

    #print("fgbdh",dis)
    disf=disf+dis
    dis=9999999

  return disf

def centroid(l):
    #print(l)
    dis=9999999
    centroid=(0,0)
    dis1=0
    a=0
    b=0
    for i in range (0,len(l)):
        a=a+l[i][0]
        b=b+l[i][1]
    if(len(l)!=0):
     centroid=(a/len(l),b/len(l))

    return centroid




def global_kmeans(l,k):
 global max_Iter
 global clusters1
 cen=[]
 cen1=[]
 cen2=[]
 init_error=9999999
 for i in range(1,k+1):
     if(i==1):
      cen.append(centroid(l))
      #print(cen)
      #I=i+1
     else:
       for j in range(0,len(l)):
          cen.append(l[j])
          #print(cen,"fdgsjcdgxhsjdvbxhj")
          cen1=kMeans(cen,i,l)
          #print(cen1)
          max_Iter=10
          err1=dist(cen1,l)
          #print("error==",err1,"error==",init_error)
          if(err1<init_error):
            clusters1=clusters
            #print("vfbdnmsfbdnms")
            init_error=err1
            cen2=cen1

          del(cen[len(cen)-1])
       init_error=9999999

       cen=cen2

 return cen2



def assign_cluster(l2,cen):
    cluster=1
    dist=9999999
    for i in range(0,len(cen)):
        dis1=(l2[0]-cen[i][0])*(l2[0]-cen[i][0])+(l2[1]-cen[i][1])*(l2[1]-cen[i][1])
        #print(dis1,cen[i],l2,i )
        if dis1<dist:

            dist=dis1
            cluster=i+1
    #print(cen)
    #print(cluster)


    return cluster

h4=0

def kMeans(cen,K,l):

    global clusters
    global h4
    h4=h4+1
    print(h4)
    #print(cen,K,"ghjkbn")
    global max_Iter
    l_len=len(l)
    l1=[]
    cen1=[]
    if max_Iter!=0:
        max_Iter=max_Iter-1
        clusters=[]
        for i in range(0,len(l)):
            clusters.append(assign_cluster(l[i],cen))
            #print(   i  )
        for j in range(0,K):
            print(clusters)
            for i in range(0,len(l)):
                if(clusters[i]==j+1):
                    l1.append(l[i])
            #print(l1)
            cen1.append(centroid(l1))
            l1=[]
    else:
       return cen
    cen1=kMeans(cen1,K,l)
    return cen1

#check wg=here k where k+1

def max(a,b):
  if(a>b):
   return a
  return b

def fastk_means(l,k):

  global clusters
  cen=[]
  cen1=[]
  init_error=9999999

  for i1 in range(0,len(l)):
      clusters.append(1)
  for i in range(1,k+1):
      if(i==1):
       cen.append(centroid(l))

      else:
        max1=0
        b=[]
        for j in range(0,len(l)):
             b.append(0)

             for j1 in range(0,len(l)):
                d=(l[j1][0]-cen[clusters[j1]-1][0])*(l[j1][0]-cen[clusters[j1]-1][0])+(l[j1][1]-cen[clusters[j1]-1][1])*(l[j1][1]-cen[clusters[j1]-1][1])
                d1=(l[j][0]-l[j1][0])*(l[j][0]-l[j1][0])+(l[j][1]-l[j1][1])*(l[j][1]-l[j1][1])
                d=d-d1
                d=max(d,0)
                b[j]=b[j]+d
                print(b[j],"    ")
                print("   ")
             if(b[j]>max1):
                max1=b[j]
                index=j
                print(index," hjnm  ")
        cen.append(l[index])
        cen=kMeans(cen,i,l)

        print(cen,"fgdhjskdbnsj")
        max_Iter=10
  return cen




l=[(1,2),(93,2),(12,5),(54,16),(6,9),(5,4),(2,7),(4,6),(10,28),(45,18)]
k=2

"""leaf=[]
def build_kdtree(points, depth=0):
    n = len(points)

    if n <= 2:
        leaf.append(points)
        return None

    axis = depth % k

    sorted_points = sorted(points, key=lambda point: point[axis])

    return {
        'point': sorted_points[int(n / 2)],
        'left': build_kdtree(sorted_points[:int(n / 2)], depth + 1),
        'right': build_kdtree(sorted_points[int(n/2) + 1:], depth + 1)
    }

b=[]
with open('cassini250.csv') as inputfile:
    results = csv.reader(inputfile)
    for row in results:
        a=[]
        a.append(float(row[0]))
        a.append(float(row[1]))
        b.append(a)
b1=np.array(b)
b2=b1.transpose()
l=[]
l=build_kdtree(b1)

nodes=[]
for i in leaf:
    sumx=0
    sumy=0
    for j in i:
        sumx+=float(j[0])
        sumy+=float(j[1])
    sumx/=len(i)
    sumy/=len(i)
    point=[]
    point.append(sumx)
    point.append(sumy)
    nodes.append(point)
"""
b=[]
with open('unbalance.csv') as inputfile:
    results = csv.reader(inputfile)
    for row in results:
        a=[]
        a.append(float(row[0]))
        a.append(float(row[1]))
        b.append(a)
b1=np.array(b)
nodes=b1

cen1=fastk_means(nodes,3)
for i in range(0,len(nodes)):
  clusters[i]=assign_cluster(nodes[i],cen1)

x_val=[]
y_val=[]
"""
for i in cen1:
    x_val.append(float(i[0]))
    y_val.append(float(i[1]))
x_val1=[]
y_val1=[]
for i in results:
    x_val1.append(float(i[0]))
    y_val1.append(float(i[1]))
"""

print(cen1)
lo=[]
x_val1=[]
y_val1=[]
#print(clusters)
for i in range(0,4):
    for j in range(0,len(clusters)):
        if(clusters[j]==i+1):
           lo.append(nodes[j])

    for k2 in lo:
        x_val1.append(float(k2[0]))
        y_val1.append(float(k2[1]))
    if(i==1):
        plt.plot(x_val1,y_val1,'yo')
    elif(i==2):
        plt.plot(x_val1,y_val1,'ro')
    elif(i==3):
        plt.plot(x_val1,y_val1,'go')
    else:
        plt.plot(x_val1,y_val1,'bo')

    lo=[]
    x_val1=[]
    y_val1=[]
plt.show()

"""plt.plot(x_val1,y_val1,'yo')
plt.plot(x_val,y_val,'ro')
plt.show()

plt.show()
"""
