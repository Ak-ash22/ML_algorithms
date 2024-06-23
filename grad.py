#Gradient Descent for Linear Regression
import numpy as np

#yhat = wx + b
#loss = (y - yhat)**2/N = mean squared error
#      = (y - (wx+b))**2/N

x = np.random.rand(10,1)
y = 10*x + 3    #Given y function with defined w and b

#Initialize Parameters
w = 0.0
b = 0.0

#Hyperparameter
learning_rate = 0.1

#Define a gradient descent function
def gradient_descent(x,y,w,b,learning_rate):
    
    #derivates of loss wrt parameters
    dldw = 0
    dldb = 0
    N = x.shape[0]
    
    for xi,yi in zip(x,y):
        dldw += 2*(yi-(w*xi+b))*(-xi)
        dldb += 2*(yi-(w*xi+b))*(-1)
    
    #Updating the paramters
    w = w - learning_rate*dldw/N    #Taking the average of total derivative
    b = b - learning_rate*dldb/N    
    
    return w,b


#Iteratively make update to the paramters until ideally u reach the predescribed w and b

for epoch in range(400):
    w,b = gradient_descent(x,y,w,b,learning_rate)
    
    yhat = w*x + b  #Predicted y values
    
    loss = np.sum((y-yhat)**2)/x.shape[0]
    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')