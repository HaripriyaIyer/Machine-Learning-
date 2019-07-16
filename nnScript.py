import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import sqrt
import pandas as pd
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1.0 / (1.0 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    

    train_data = final_trainData = np.zeros((50000,28*28), dtype=np.int)
    train_label = final_trainLabel = np.zeros(50000, dtype=np.int)
    validation_data = val_data = np.zeros((10000,28*28), dtype=np.int)
    validation_label = val_dataLabel = np.zeros(10000, dtype=np.int)
    test_data = final_testData = np.zeros((10000,28*28), dtype=np.int)
    test_label = final_testLabel = np.zeros(10000, dtype=np.int)
    
    start = 0
    test_start = 0
    val_start = 0
    
    for d_key in mat:
        # Split into train and test set.
        if 'train' in d_key:
    #        print('d_key: ',d_key)
            label = d_key[-1]       # Extract the class label 0 to 9
            trainData = mat.get(d_key)
            len_trainData = len(trainData)
    #        print('len_trainData: ',len_trainData)
            sample = np.random.choice(len_trainData, 1000, replace = False)  # Create a sample for validation set
           
            
            label_vec = np.full(shape = len_trainData, 
                                fill_value = label , 
                                dtype = np.int)         # Create ground truth vector: Y
            
            # Copy everything except SAMPLE index values. 
            # SAMPLE index values are copied into validation set.
            final_trainData[start:start+len_trainData - 1000] = np.delete(trainData, sample, axis=0) 
            final_trainLabel[start:start+len_trainData - 1000] = np.delete(label_vec, sample, axis=0)
    #        print('final_train index: ',start+len_trainData - 1000)
            start += len_trainData - 1000  
    #        print('final_train index: ',start)
            
            val_data[val_start:val_start+1000] = trainData[sample[:],]
            val_dataLabel[val_start:val_start+1000] = label_vec[sample[:],]
    #        print('validation index: ',val_start +1000)
            val_start += 1000
            
        elif 'test'in d_key:
            label = d_key[-1]
    #        print('label: ', label)
            testData = mat.get(d_key)
            len_testData = len(testData)
    #        print('length: ', len_testData)
            label_vec = np.full(shape = len_testData, 
                                fill_value = label , 
                                dtype = np.int)
            
            final_testData[test_start:test_start+len_testData] = testData
            final_testLabel[test_start:test_start+len_testData] = label_vec
            test_start += len_testData
            
    train_data = np.double(final_trainData)
    train_data = train_data / 255.0

    validation_data = np.double(val_data)
    validation_data = validation_data / 255.0

    test_data = np.double(final_testData)
    test_data = test_data / 255.0

    # Feature selection

    total_input_data = np.row_stack((train_data,validation_data,test_data))
    zero_feat = np.all(total_input_data == 0, axis=0) # Check which columns have value '0' throughout (returns boolean)
    total_input_data = total_input_data[:,~zero_feat] # Eliminate zero-valued columns
   
    selected_features = list(set(range(0,784)) - set(zero_feat))
    pickle.dump((selected_features),open('selected_features.pickle','wb'))
    selected_features = pickle.load(open('selected_features.pickle','rb'))
        
    train_data = total_input_data[0:len(train_data),:] 
    validation_data = total_input_data[len(train_data):(len(train_data) + len(validation_data)) ,:]
    test_data = total_input_data[(len(train_data) + len(validation_data)) : (len(test_data) + len(validation_data) + len(train_data)),:]

            
#     train_data = final_trainData
    train_label = final_trainLabel        
#     test_data = final_testData 
    test_label = final_testLabel 
#     validation_data = val_data
    validation_label = val_dataLabel

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:(n_input + 1) * n_hidden].reshape((n_hidden,(n_input + 1)))  # Weights from Input to Hidden
    w2 = params[(n_input + 1) * n_hidden:].reshape((n_class,(n_hidden + 1)))   # Weights from Hidden to Output
    obj_val = 0
    
    num_feat = training_data.shape[1]
    num_obs = training_data.shape[0]
    num_w_ih = w1.shape[0]
    num_w_ho = w2.shape[0]
    
    # Feedforward
    X = np.column_stack((training_data, np.ones((num_obs)))) # add bias term to training_data
    xw = np.dot(X,w1.T)   # Input to Hidden Layer
    sig_h = sigmoid(xw) # Sigmoid of hidden layer output
    h_out = np.column_stack((sig_h, np.ones((sig_h.shape[0]))))# Add Bias term to hidden layer output
    hw = np.dot(h_out,w2.T) # Input to Output Layer
    sig_o = sigmoid(hw)   # Sigmoid of Output Layer
    
    # Creating Truth table
    true_y = np.zeros((num_obs, n_class)) 
    for i in range(num_obs):
        true_y[i][training_label[i]] = 1
        
    loss = -(true_y - sig_o)
        
    ## Backpropagation
    # Objective_Func = squared error loss. 
    grad_out_hid  = np.zeros((num_w_ho, num_w_ih+1))
    grad_hid_inp= np.zeros((num_w_ih, num_obs+1))
    
    #Error from Output to Hidden Layer
    dJ_out = sig_o * (1 - sig_o)
    err_out_hid = loss * dJ_out
    grad_out_hid = np.dot((np.transpose(err_out_hid)),h_out) #dJ/dW_of_hidden
    
    #Error from Hidden to Input
    err_hid_inp1 = (np.dot(err_out_hid,w2)) #dJ/dW_of_input
    err_hid_inp2 = ((1 - h_out) * h_out) #Derivative of sigmoid at hidden layer
    err_hid_inp = err_hid_inp1 * err_hid_inp2
    grad_hid_inp = np.dot((err_hid_inp.T),X)
    grad_hid_inp= grad_hid_inp[0:n_hidden,:]
    
    grad_hid_inp = (grad_hid_inp + (lambdaval * w1)) / num_obs
    grad_out_hid = (grad_out_hid + (lambdaval * w2)) / num_obs
    
    obj_val_1 = np.multiply(true_y,(np.log(sig_o)))
    obj_val_2 = np.multiply((1-true_y),(np.log(1-sig_o)))
    obj_val = np.sum(obj_val_1 + obj_val_2)
    obj_val = -obj_val
    obj_val = obj_val + ((lambdaval / 2)  * (np.sum(w1*w1) + np.sum(w2*w2)));
    obj_val = obj_val / num_obs;
    
    obj_grad = np.concatenate((grad_hid_inp.flatten(), grad_out_hid.flatten()),0)


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    labels = np.array([])
    num_obs = data.shape[0]
    num_feat = data.shape[1] 
    
    
    X = np.column_stack((data, np.ones((num_obs)))) # add bias term to train_data
    xw = np.dot(X,w1.T)   # Input to Hidden Layer
    sig_h = sigmoid(xw) # Sigmoid of hidden layer output
    h_out = np.column_stack((sig_h, np.ones((sig_h.shape[0]))))# Add Bias term to hidden layer output
    hw = np.dot(h_out,w2.T) # Input to Output Layer
    sig_o = sigmoid(hw)   # Sigmoid of Output Layer
    
    labels = np.argmax(sig_o,axis = 1) #Maximum probability value taken for prediction
    labels = labels[:, np.newaxis]

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

import time
hyper_params = ['lambda','Hidden','Time_Taken','Train_Accuracy','Validation_Accuracy','Test_Accuracy']
results = pd.DataFrame(columns = hyper_params).astype({'lambda' : 'int8','Hidden' : 'int8'})

lambdaval = [0, 10, 20, 30, 40, 50, 60]
num_hidden_layers = [4, 8, 12, 16, 20]
for i in lambdaval:
    for j in num_hidden_layers:
        
        s = time.time()
        n_input = train_data.shape[1]; # number of input nodes = num of features
        n_hidden = j; # number of hidden nodes
        n_class = 10; # number of output nodes

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden);
        initial_w2 = initializeWeights(n_hidden, n_class);

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

        # set the regularization hyper-parameter
        lambdaval = i;
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        #Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter' : 50}    # Preferred value.



        #objv, obgg = nnObjFunction(initialWeights, args)

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
        params = nn_params.get('x')
        #In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        #and nnObjGradient. Check documentation for this function before you proceed.
        #nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


        #Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        predicted_label = nnPredict(w1,w2,train_data)
       
        #find the accuracy on Training Dataset

        print('\n Training set Accuracy:' + str(100*np.mean(predicted_label==train_label.reshape(train_label.shape[0],1))) + '%')
        train_accuracy = 100*np.mean(predicted_label==train_label.reshape(train_label.shape[0],1))

        predicted_label = nnPredict(w1,w2,validation_data)

        #find the accuracy on Validation Dataset

        print('\n Validation set Accuracy:' + str(100*np.mean(predicted_label==validation_label.reshape(validation_label.shape[0],1))) + '%')
        validation_accuracy = 100*np.mean(predicted_label==validation_label.reshape(validation_label.shape[0],1))


        predicted_label = nnPredict(w1,w2,test_data)

        #find the accuracy on Validation Dataset

        print('\n Test set Accuracy:' +  str(100*np.mean(predicted_label==test_label.reshape(test_label.shape[0],1))) + '%')
        test_accuracy = 100*np.mean(predicted_label==test_label.reshape(test_label.shape[0],1))
        
        e=time.time()
        time_taken = e-s
        
        results = results.append({'lambda':i,'Hidden':j,'Time_Taken':time_taken, 'Train_Accuracy':train_accuracy,'Validation_Accuracy':validation_accuracy,'Test_Accuracy':test_accuracy}, ignore_index=True)
    print(results)
        


# In[Trace_1]:

lambdaval_values = results['lambda']
lambda_time = results['Time_Taken']

plt.scatter(lambdaval_values,lambda_time)
plt.show()


# In[Trace_2]:

hidden_values = results['Hidden']
hidden_time = results['Time_Taken']

plt.scatter(hidden_values,hidden_time)
plt.show()


# In[Trace_3]:

lambda_mean = results.groupby(results['lambda']).agg({"Time_Taken":np.mean })
print(lambda_mean)
lambda_mean.plot.bar()
plt.show()


# In[Trace_4]:

hidden_mean = results.groupby("Hidden").agg({"Time_Taken":np.mean })
print(hidden_mean)
hidden_mean.plot.bar()
plt.show()


# In[Trace_5]:

test_accuracy_mean = results.groupby("lambda").agg({"Test_Accuracy" : np.mean})
print(test_accuracy_mean)
test_accuracy_mean.plot()
plt.show()


# In[Trace_6]:

accuracy_mean = results.groupby("lambda").agg({"Train_Accuracy" : np.mean, "Validation_Accuracy" : np.mean, "Test_Accuracy" : np.mean})
print(accuracy_mean)
accuracy_mean.plot()
plt.show()


# In[Trace_7]:

hidden_accuracy_mean = results.groupby("Hidden").agg({"Train_Accuracy" : np.mean, "Validation_Accuracy" : np.mean, "Test_Accuracy" : np.mean})
print(hidden_accuracy_mean)
hidden_accuracy_mean.plot()
plt.show()


# In[Trace_8]:

hidden_accuracy_mean


# In[Trace_9]:

accuracy_mean


# In[Trace_10]:

pickle.dump((n_hidden,w1,w2,lambdaval),open('params.pickle','wb'))
param_pickle = pickle.load(open('params.pickle','rb'))
print(param_pickle)


