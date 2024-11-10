import numpy as np
import pandas as pd


#Ham sigmoid (activation func)
def sigmoid(x):
    return 1/(1+np.exp(-x))


#Vi phan ham sigmoid theo x
def sigmoid_derivaticve(x):
    return x*(1-x)


#Khoi tao lop neural retwork 
class NeuralNetwork:
    def __init__(self,layers,alpha=0.1):
        self.layers=layers #Layers la 1 list tu dinh nghia ve so lop, so node moi lop ([2,2,1])
        self.alpha=alpha # He so hoc tu dinh nghia
        self.W=[] #khoi tao ma tran cac trong so Wij cua cac lien ket giua cac lop hidden layers
        self.b=[] #khoi tao cac tham so b1,b2,b3,... o cac lop hidden layers 

        #Bat dau khoi tao cac tham so o moi lien ket va nnut cuar hidden layers
        for i in range(0, len(layers)-1):
            w_=np.random.randn(layers[i],layers[i+1]) # tao ra 1 ma tran random cac so co layers[i] hang, layers [i+1] cot ung voi tung lien ket sau
            b_=np.zeros((layers[i+1],1)) # tao ra 1 cot toan 0 cho cac bias cua tung lop gan voi cac lop sau
            self.W.append(w_/layers[i]) #them cac trong so vao list chua cac W va chia cho so node co trong lop do ---> can doi trong so trong lop de dat do can bang trong viec huan luyen mo hinh
            self.b.append(b_) # them cac bias vao trong list chua cac bias


    #Tom tat mo hinh neral network
    def __repr__(self):
        return "Neural network [{}]".format ("-".join(str(l) for l in self.layers))
    
    #Train mo hinh voi du lieu co san
    def fit_partial(self,x,y):
        A=[x] # list xac dinh cac dau ra cuar cac noron trong tung lop, khoi tao la dau vao

        #feedforward
        out =A[-1]
        for i in range(0,len(self.layers)-1):
            out = sigmoid(np.dot(out,self.W[i])+self.b[i].T) # to hop tuyen tinh cac weight va dau ra cac lop truoc +bias va cho vao ham kich hoat
            A.append(out) # sau khi xac dinh dau ra duoc cho vao list A


        #back propagation
        y=y.reshape(-1,1) # reshape output chi co 1 cot 
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))] # vi phan ham loss theo gia tri du doan
        dW = []
        db = []
        for i in reversed(range (0,len(self.layers)-1)):
            dw_=np.dot((A[i]).T,dA[-1]* sigmoid_derivaticve(A[i+1]))
            db_=(np.sum(dA[-1]*sigmoid_derivaticve(A[i+1],0)).reshape(-1,1))
            dA_=np.dot(dA[-1]* sigmoid_derivaticve(A[i+1]),self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
        # tinh gradient cua trong so va bias theo tung lop dua vao chain rule
        # luu tru gradient trong 2 list     

        dW=dW[::-1] # dao lai tu cuoi len dau thanh dau xuong cuoi
        db=db[::-1]

        #Gradient descent 
        for i in range ( 0, len(self.layers)-1):
            self.W[i]=self.W[i]-self.alpha*dW[i]
            self.b[i]=self.b[i]-self.alpha*db[i]


    def fit(self,X,y,epochs=20, verbose=10):  # X: data input, y data output co san
        for epoch in range (0, epochs):
            self.fit_partial(X,y)
            if epoch%verbose==0:
                loss=self.calculate_loss(X,y)
                print("Epoch {}, loss {}".format(epoch,loss))
    
    def predict(self,X):
        for i in range(0,len(self.layers)-1):
            X= sigmoid(np.dot(X,self.W[i])+self.b[i].T)
        return X
    
    def calculate_loss(self,X,y):
        y_predict=self.predict(X)
        return -(np.sum(y*np.log(y_predict)+(1-y)*np.log(1-y_predict)))


