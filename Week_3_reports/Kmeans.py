from __future__ import print_function
import numpy as np ## ma tran
import matplotlib.pyplot as plt # hien thi du lieu
from scipy.spatial.distance import cdist  # khoang cach giua cac tap diem
np.random.seed(11)


# Tao du lieu 
means=[[2,2],[8,3],[3,6]] # các điểm trung tâm ban đầu
cov=[[1,0],[0,1]]
N=500 # so diem du lieu
X0=np.random.multivariate_normal(means[0],cov,N) # tạo tập hợp điểm xung quanh tâm 
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
        centers[k,:]=np.mean(Xk,axis=0)## trung binh toa do Xk
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
        labels.append(kmeans_assign_labels(X,centers[-1]))##dán nhãn cho các điểm dựa trên tọa độ tâm các cluster
        new_centers=kmeans_update_centers(X,labels[-1],K) ## update các tâm nhóm dựa trên kết quả vừa chia 
        if has_converged(centers[-1],new_centers):  ##  nếu hội tụ thì dừng
            break
        centers.append(new_centers) # thêm vào lịch sử các tâm 
        it+=1
    return (centers,labels,it) # trả về tọa độ tâm, nhãn của từng điểm và số lần thực hiện 

(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])
print(it)

kmeans_display(X, labels[-1])
