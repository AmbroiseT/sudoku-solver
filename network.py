import numpy as np
import cv2
import scipy
import scipy.optimize

train_size = 9
image_size = 28
image_new_size = 10


def sigmoid(mat):
    return 1/(1+np.exp(-mat))

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z));

def predict(Theta, input_layer_size, hidden_layer_size, num_labels, X):
    theta0 = np.array(Theta[0:(input_layer_size+1)*hidden_layer_size])
    theta1 = np.array(Theta[(input_layer_size+1)*hidden_layer_size:])
    theta0 = theta0.reshape((input_layer_size+1, hidden_layer_size))
    theta1 = theta1.reshape((hidden_layer_size+1, num_labels))
    m = X.shape[0]
    X = np.hstack((np.ones([m,1]),X))
    
    a1 = sigmoid(X.dot(theta0))
    
    #a1 = np.vstack((np.ones([1,a1.shape[1]]),a1)) 
    a1 = np.append(np.ones([a1.shape[0], 1]),a1, 1) #Add bias

    a2 = sigmoid(a1.dot(theta1))
    return a2

def load_x_for_val(val, m):
    img = cv2.imread("data/mnist_train"+str(val)+".jpg", cv2.IMREAD_GRAYSCALE)
    i = np.array(range(train_size))
    product = np.transpose([np.tile(i, len(i)), np.repeat(i, len(i))])
    np.random.shuffle(product)
    X = np.ones((m, image_new_size**2))
    
    for index, position in enumerate(product[1:m]):
        x1, y1, x2, y2 = position[0]*image_size, position[1]*image_size, (position[0]+1)*image_size, (position[1]+1)*image_size
        car = img[y1:y2, x1:x2]
        car = cv2.resize(car, (image_new_size, image_new_size))
        X[index] = car.flatten()
    return X

def nn_costfunction(nn_params, *args):
    input_layer_size, hidden_layer_size, num_labels, X, y, lam = args[0], args[1], args[2], args[3], args[4], args[5]
    
    theta0 = np.array(nn_params[0:(input_layer_size+1)*hidden_layer_size])
    theta1 = np.array(nn_params[(input_layer_size+1)*hidden_layer_size:])
    theta0 = theta0.reshape((input_layer_size+1, hidden_layer_size))
    #print((hidden_layer_size+1)* num_labels)
    theta1 = theta1.reshape((hidden_layer_size+1, num_labels))

    m = X.shape[0]
    X = np.hstack((np.ones([m,1]),X))
    
    a1 = sigmoid(X.dot(theta0))
    
    #a1 = np.vstack((np.ones([1,a1.shape[1]]),a1)) 
    a1 = np.append(np.ones([a1.shape[0], 1]),a1, 1) #Add bias
    a2 = sigmoid(a1.dot(theta1))
    
    
    J = 0
    for row in range(m):
        for k in range(num_labels):
            valueY = y[row][k]
            J = J + (-(valueY)*np.log(a2[row][k]) - (1-valueY)*np.log(1-a2[row][k]))
    J = (1/float(m))*J
    J = J + (lam/float(2*m)) *(np.sum(np.square(theta0[:][2:]))+np.sum(np.square(theta1[:][2:])))
    print J
    return J
    
def nn_gradient(nn_params, *args):
    input_layer_size, hidden_layer_size, num_labels, X, y, lam = args[0], args[1], args[2], args[3], args[4], args[5]
    

    theta0 = np.array(nn_params[0:(input_layer_size+1)*hidden_layer_size])
    theta1 = np.array(nn_params[(input_layer_size+1)*hidden_layer_size:])
    theta0 = theta0.reshape((input_layer_size+1, hidden_layer_size))
    theta1 = theta1.reshape((hidden_layer_size+1, num_labels))
    m = X.shape[0]
    X = np.hstack((np.ones([m,1]),X))
    
    a1 = X
    z2 = X.dot(theta0) 
    
    a2 = sigmoid(z2)
    #a1 = np.vstack((np.ones([1,a1.shape[1]]),a1)) 
    a2 = np.append(np.ones([a1.shape[0], 1]),a2, 1) #Add bias

    z3 = a2.dot(theta1)
    h = sigmoid(z3)

    Theta0_grad = np.zeros(theta0.shape)
    Theta1_grad = np.zeros(theta1.shape)
    for t in range(m):
        
        a_1 = a1[t,:].reshape(1, input_layer_size+1)        
        z_2 = z2[t,:].reshape(1, hidden_layer_size)
        a_2 = a2[t,:].reshape(1, hidden_layer_size+1)
        
        valueH = h[t, :].reshape(1, num_labels)
        valueY = y[t].reshape(1, num_labels)
        delta3 = (valueH - valueY)
        delta2 = np.dot(theta1,delta3.T)*(np.append(np.array([[1]]), sigmoidGradient(z_2), 1).T)
        
        delta2 = delta2[1:]
             
        
        Theta0_grad += (np.dot(delta2,a_1)).T
        Theta1_grad += np.dot(a_2.T, delta3)
        
    #Regularize gradient
    Theta1_grad = (1/float(m))*Theta1_grad + (float(lam)/m)*theta1
    Theta0_grad = (1/float(m))*Theta0_grad + (float(lam)/m)*theta0

    grad = np.append(np.ndarray.flatten(Theta0_grad), np.ndarray.flatten(Theta1_grad))
    
    return grad

def grad_check(costfunc, theta, args):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 0.0001
    for p in range(theta.shape[0]):
        perturb[p] = e
        loss1 = costfunc(theta-perturb, *args)
        loss2 = costfunc(theta+perturb, *args)
        numgrad[p] = (loss2-loss1)/(2*e)
        perturb[p] = 0
    return numgrad

def load_every_xy(m):
    contentX = []
    contentY = []
    for i in range(10):
        x = load_x_for_val(i, m)
        y = [0]*10
        y[i] = 1
        y = np.array([y]*m)
        contentX.append(x)
        contentY.append(y)
    X = np.concatenate(contentX)
    Y = np.concatenate(contentY)
    return X, Y

def evaluate_accuracy(m, theta0, theta1):
    Xval, yval = load_every_xy(m)
    good = 0
    for i in range(m*10):
        prediction = predict(Xval[i], theta0, theta1)
        print (prediction)
        answer = np.where(yval[i]==1)
        print (answer[0][0])
        if prediction == answer[0][0]:
            good+=1
    return float(good)/(m*10)
    

def load_every_xy_onevsall(m, val):
    contentX = []
    contentY = []
    for i in range(10):
        x = load_x_for_val(i, m)
        #y = 1 if i==val else 0
        #y = np.array([y]*m)
        y = [0]*10
        y[i] = 1 
        y = np.array([y]*m)    
        contentX.append(x)
        contentY.append(y)
    X = np.concatenate(contentX)
    Y = np.concatenate(contentY)
    #Y = np.reshape(Y, (m*10, 1))
    return X, Y

#Dimension of the hidden layer:
hidden_dim = 25
#X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
#X = load_x_for_val(3, 10)
#y = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]*10)
oX, oy = load_every_xy_onevsall(20, 3)
X = oX[:100]
y = oy[:100]
Xtest = oX[100:]
ytest = oy[100:]
#X, y = load_every_xy(20)
#y = np.array([[0,1,1,0]]).T

epsilon_init = 0.12
theta0 = epsilon_init*2*np.random.random((X.shape[1]+1,hidden_dim))-epsilon_init
theta1 = epsilon_init*2*np.random.random((hidden_dim+1,y.shape[1]))-epsilon_init

#print theta1.shape
Theta = np.concatenate([np.reshape(theta0, ((X.shape[1]+1)*hidden_dim, 1)), np.reshape(theta1, ((hidden_dim+1)*y.shape[1], 1))])
print(nn_costfunction(Theta, X.shape[1], theta0.shape[1], y.shape[1], X, y, 0.5))
#print(nn_gradient(Theta, X.shape[1], theta0.shape[1], y.shape[1], X, y, 0)-grad_check(nn_costfunction, Theta, args=(X.shape[1], theta0.shape[1], y.shape[1], X, y, 0)))
#print(grad_check(nn_costfunction, Theta, args=(X.shape[1], theta0.shape[1], y.shape[1], X, y, 0.1)))



print "Over"
optTheta = scipy.optimize.fmin_cg(nn_costfunction, Theta, maxiter=400, args=( X.shape[1], theta0.shape[1], y.shape[1], X, y, 1), fprime=nn_gradient)

'''
for j in xrange(300):
    a1 = sigmoid(np.dot(X, theta0))
    a2 = sigmoid(np.dot(a1, theta1))
    delta2 = (y-a2)*(a2*(1-a2))
    delta1 = delta2.dot(theta1.T)*(a1*(1-a1))
    theta1 += a1.T.dot(delta2)
    theta0 += X.T.dot(delta1)
'''

#Xtest, ytest = X, y
Xtest, ytest = load_every_xy_onevsall(10, 3)
result = predict(optTheta, X.shape[1], theta0.shape[1], y.shape[1], Xtest)
good = 0
total = Xtest.shape[0]
for index, pred in enumerate(result):    
    #print("res = {}".format(pred))
    print "Prediction n {} : ".format(index)    
    print np.argmax(pred)
    print np.argmax(ytest[index])
    if np.argmax(pred) == np.argmax(ytest[index]):
        good+=1
    
print "Accuracy in validation set 10 iter : {}".format(float(good)/total)


'''
accuracy = evaluate_accuracy(10, theta0, theta1)
print( accuracy)
'''
'''
toPredict = np.array([ [1,1,1] ])
print predict(toPredict, theta0, theta1)
'''