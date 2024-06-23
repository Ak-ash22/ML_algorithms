import numpy as np

#Motivation --- Exploding/ Vanishing gradient problem ; computation time  

#Toy Data
X = np.random.rand(100,1)

#Defining Y with pre-defined parameters
theta1 = 2
theta2 = 3
#noise = np.random.rand(100,1)
Y = theta1 + theta2 * X

#Redefining the Data to make it covenient for mathematics later
X_b = np.c_[np.ones([100,1]),X]

#parameters
n_epoch = 20
t0 = 5
t1 = 10

N = np.shape(X)[0]
print(N)

#hyperparameter
def learning_rate(t):
    return t0/(t+t1)    #Or  = 1/t

#Loss function -- MSE = (y - yhat)**2/N
##gradient = derivative of theta w.r.t MSE Loss function

theta = np.random.randn(2,1)   #For theta1 and theta 2 parameter

def stochastic_descent(x,y,epoch,theta,learning_rate):
    
    dldt = 2*x.T.dot(x.dot(theta) - y)
    eta = learning_rate(epoch+1)
    
    #Updating Parameters
    theta = theta - eta*dldt
    return theta

for epoch in range(n_epoch):
    
    for i in range(N):
        random_index = np.random.randint(N)
        
        #We do Classic single data point SGD algorithm
        xi  = X_b[random_index:random_index+1]
        yi = Y[random_index:random_index+1]
        
        theta = stochastic_descent(xi,yi,epoch,theta,learning_rate)
        
    yhat = np.dot(X_b,theta)
    loss = np.sum((Y-yhat)**2)/N    
    print(f'{epoch} loss is {loss}, parameters:theta = {theta}')
        
print(np.shape(theta))        