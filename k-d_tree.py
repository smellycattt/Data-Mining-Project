import numpy
import csv
import matplotlib.pyplot as plt
import pprint

#change leafsize according to need, i think it was 2000
def kdtree( data, leafsize=10 ):
    ndim = data.shape[0]
    ndata = data.shape[1]

    # find bounding hyper-rectangle
    hrect = numpy.zeros((2,data.shape[0]))
    hrect[0,:] = data.min(axis=1)
    hrect[1,:] = data.max(axis=1)

    # create root of kd-tree
    idx = numpy.argsort(data[0,:], kind='mergesort')
    data[:,:] = data[:,idx]
    splitval = data[0,ndata/2]

    left_hrect = hrect.copy()
    right_hrect = hrect.copy()
    left_hrect[1, 0] = splitval
    right_hrect[0, 0] = splitval

    tree = [(None, None, left_hrect, right_hrect, None, None)]

    stack = [(data[:,:ndata/2], idx[:ndata/2], 1, 0, True),
             (data[:,ndata/2:], idx[ndata/2:], 1, 0, False)]

    # recursively split data in halves using hyper-rectangles:
    while stack:

        # pop data off stack
        data, didx, depth, parent, leftbranch = stack.pop()
        ndata = data.shape[1]
        nodeptr = len(tree)

        # update parent node

        _didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]

        tree[parent] = (_didx, _data, _left_hrect, _right_hrect, nodeptr, right) if leftbranch \
            else (_didx, _data, _left_hrect, _right_hrect, left, nodeptr)

        # insert node in kd-tree

        # leaf node?
        if ndata <= leafsize:
            _didx = didx.copy()
            _data = data.copy()
            leaf = (_didx, _data, None, None, 0, 0)
            tree.append(leaf)

        # if not a leaf, split the data in two      
        else:
            splitdim = depth % ndim
            idx = numpy.argsort(data[splitdim,:], kind='mergesort')
            data[:,:] = data[:,idx]
            didx = didx[idx]
            nodeptr = len(tree)
            stack.append((data[:,:ndata/2], didx[:ndata/2], depth+1, nodeptr, True))
            stack.append((data[:,ndata/2:], didx[ndata/2:], depth+1, nodeptr, False))
            splitval = data[splitdim,ndata/2]
            if leftbranch:
                left_hrect = _left_hrect.copy()
                right_hrect = _left_hrect.copy()
            else:
                left_hrect = _right_hrect.copy()
                right_hrect = _right_hrect.copy()
            left_hrect[1, splitdim] = splitval
            right_hrect[0, splitdim] = splitval
            
            # append node to tree
            tree.append((None, None, left_hrect, right_hrect, None, None))


    return tree

b=[]
with open('synth.te.csv') as inputfile:
    results = csv.reader(inputfile)
    for row in results:
        a=[]
        a.append(float(row[0]))
        a.append(float(row[1]))
        b.append(a)
b1=numpy.array(b)
b2=b1.transpose()
l=kdtree(b2,400)
nodes=[]
for i in l:
    if(i[2]==None and i[3]==None and i[4]==0 and i[5]==0):
        sumx=0
        sumy=0
        for j in i[1]:
            sumx+=float(j[0])
            sumy+=float(j[1])
        sumx/=len(i[1][0])
        sumy/=len(i[1][1])
        point=[]
        point.append(sumx)
        point.append(sumy)
        nodes.append(point)
print nodes

pp=pprint.PrettyPrinter(indent=4)
pp.pprint(l)
x_val=[]
y_val=[]
for i in b:
    x_val.append(float(i[0]))
    y_val.append(float(i[1]))
cent_x=[]
cent_y=[]
for i in nodes:
    cent_x.append(float(i[0]))
    cent_y.append(float(i[1]))
plt.plot(x_val,y_val,'ro')
plt.plot(cent_x,cent_y,'bo')	
plt.show()
