from __future__ import print_function
import numpy as np ## ma tran
import matplotlib.pyplot as plt # hien thi du lieu
from scipy.spatial.distance import cdist  # khoang cach giua cac tap diem
np.random.seed(11)


# Tao du lieu 

means=[[2,2],[8,3],[3,6]]
cov=[[1,0],[0,1]]
N=500 # so diem du lieu
X0=np.random.multivariate_normal(means[0],cov,N)
X1=np.random.multivariate_normal(means[1],cov,N)
X2=np.random.multivariate_normal(means[2],cov,N)
X= np.concatenate((X0,X1,X2),axis=0)
K=3 # So cum du dinh se chia
original_label=np.asarray([0]*N+[1]*N+[2]*N).T

def kmeans_display(X,label):
    K=np.amax(label)+1
    X0=X[label==0,:]
    X1=X[label==1,:]
    X2=X[label==2,:]

    plt.plot(X0[:,0],X0[:,1],'b^',markersize=4,alpha=.8)
    plt.plot(X1[:,0],X1[:,1],'go',markersize=4,alpha=.8)
    plt.plot(X2[:,0],X2[:,1],'rs',markersize=4,alpha=.8)
    plt.axis('equal')
    plt.plot()
    plt.show()


kmeans_display(X,original_label)


#khoi tao kmeasn ban dau
def kmeasn_init_centers(X,k):
    return X[np.random.choice(X.shape[0],k,replace=False)]
    #Chon k hang trong X lam centers
def kmeans_assign_labels(X,centers):
    #tinh khoang cach giua data va centers
    D=cdist(X,centers)
    return np.argmin(D,axis=1) # tra lai khoang cach be nhat

def kmeans_update_centers(X,labels,K):
    centers=np.zeros((K,X.shape[1]))
    for k in range (K):
        Xk=X[labels==k,:]
        centers[k,:]=np.mean(Xk,axis=0)
    return centers

def has_converged(centers,new_centers):
     # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))



def kmeans(X,K):
    centers=[kmeasn_init_centers(X,K)]
    labels=[]
    it=0
    while True:
        labels.append(kmeans_assign_labels(X,centers[-1]))
        new_centers=kmeans_update_centers(X,labels[-1],K)
        if has_converged(centers[-1],new_centers):
            break
        centers.append(new_centers)
        it+=1
    return (centers,labels,it)

(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])
