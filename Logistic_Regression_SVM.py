import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd

errorml = []
errorbl = []

def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, y_n = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    w = initialWeights.reshape((n_features + 1),1)    
    x = np.hstack((np.ones((n_data,1)),train_data))
    theta_n = sigmoid(np.dot(x,w))

    error = y_n * np.log(theta_n) + (1.0 - y_n)*np.log(1.0 - theta_n)
    error = -1.0 * np.sum(error)
    error = error / n_data

    error_grad = (theta_n - y_n)*x
    error_grad = np.sum(error_grad, axis=0)
    error_grad = error_grad/n_data
    
    errorbl.append(error)

    
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    n_data = data.shape[0];
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_features = data.shape[1]
    data = np.hstack((np.ones((n_data,1)),data))
    P = sigmoid(np.dot(data, W)) 

    label = np.argmax(P,axis = 1) 
    label = label.reshape((n_data,1))

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    W = params.reshape(n_feature+1,n_class)
    data = np.hstack((np.ones((n_data,1)),train_data))
    theta = np.zeros((n_data,n_class))

    for i in range(n_class):
        theta[:,i] = np.exp(np.dot(np.transpose(W)[i,:],np.transpose(data)))
    tot = np.sum(theta,1)
    for i in range(n_class):
        theta[:,i] = np.divide(theta[:,i],tot)

    logtheta = np.log(theta)
    error = 0
    for i in range(n_data):
        for j in range(n_class):
            error = error + labeli[i][j]*logtheta[i][j]

    error = error*(-1.0/n_data)
    
    theta_labeli = theta - labeli
    error_grad = np.dot(np.transpose(theta_labeli),data)/float(n_data)
    error_grad = np.transpose(error_grad).flatten()

    errorml.append(error)
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix
    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]
    n_feature = data.shape[1]

    W = W.reshape(n_feature+1,n_class)
    x = np.hstack((np.ones((n_data,1)),data))
    theta = np.zeros((n_data,n_class))

    for i in range(n_class):
        theta[:,i] = np.exp(np.dot(np.transpose(W)[i,:],np.transpose(x)))
    tot = np.sum(theta,1)
    for i in range(n_class):
        theta[:,i] = np.divide(theta[:,i],tot)

    label = np.argmax(theta,axis = 1)   
    label = label.reshape((n_data,1))

    return label
   

def plot_confusion_matrix(cls_true,cls_pred,num_classes):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    #cls_true = data.test.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
plot_confusion_matrix(train_label, predicted_label,10)


# Find the accuracy on Validation Dataset
predicted_label_v = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_v == validation_label).astype(float))) + '%')
plot_confusion_matrix(validation_label, predicted_label_v,10)

# Find the accuracy on Testing Dataset
predicted_label_t = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_t == test_label).astype(float))) + '%')
plot_confusion_matrix(test_label, predicted_label_t,10)

plt.plot(pd.DataFrame(errorbl))
	
"""
Script for Extra Credit Part
"""

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
plot_confusion_matrix(train_label, predicted_label_b,n_class)

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
plot_confusion_matrix(validation_label, predicted_label_b,n_class)

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
plot_confusion_matrix(test_label, predicted_label_b,n_class)

pd.DataFrame(errorml).plot()

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

train_data_svm=np.ones((10000,715))
train_label_svm=np.ones((10000,1))
randset=np.random.choice(50000,10000, replace = False)
i=0
for k in randset:
  train_data_svm[i]=train_data[k]
  train_label_svm[i]=train_label[k]
  i=i+1

# linear kernel 
  
clf_linear = SVC(kernel='linear'); 
clf_linear.fit(train_data_svm, train_label_svm.ravel())
lktrain = clf_linear.score(train_data,train_label.ravel())
lkvalidate = clf_linear.score(validation_data,validation_label.ravel())
lktest = clf_linear.score(test_data,test_label.ravel())

print("########################################Result for Linear Kernel SVM Model########################################")
print("Training set Accuracy : ",str(lktrain))
print("Validation Set Accuracy: ",str(lkvalidate))
print("Test Set Accuracy : ",str(lktest))

#RBF gamma default

clf_rbf = SVC(kernel='rbf', gamma='auto') 
clf_rbf.fit(train_data_svm, train_label_svm.ravel())
rbftrain = clf_linear.score(train_data,train_label.ravel())
rbfvalidate = clf_linear.score(validation_data,validation_label.ravel())
rbftest = clf_linear.score(test_data,test_label.ravel())

print("########################################Result for RBF with gamma default SVM Model########################################")
print("Training set Accuracy : ",str(rbgtrain))
print("Validation Set Accuracy: ",str(rbfvalidate))
print("Test Set Accuracy : ",str(rbftest))

#RBF gamma 1  
  
clf_rbf_g1 = SVC(kernel='rbf' , gamma = 1.0)
clf_rbf_g1.fit(train_data_svm, train_label_svm.ravel())
rbfgtrain = clf_rbf_g1.score(train_data,train_label.ravel())
rbfgvalidate = clf_rbf_g1.score(validation_data,validation_label.ravel())
rbfgtest = clf_rbf_g1.score(test_data,test_label.ravel())

print("########################################Result for RBF with gamma = 1 SVM Model########################################")
print("Training set Accuracy : ",str(rbfgtrain))
print("Validation Set Accuracy: ",str(rbfgvalidate))
print("Test Set Accuracy : ",str(rbfgtest))
  
  
#RBF with varying value of C  

clf_c_df = pd.DataFrame(columns=['C', 'Training', 'Validation','Test'])

s=[]
l = [1,10,20,30,40,50,60,70,80,90,100]
for i in range(len(l)):
  clf_c = SVC(C = l[i],kernel='rbf')
  clf_c.fit(train_data_svm, train_label_svm.ravel())
  s.append([l[i], clf_c.score(train_data,train_label.ravel()),clf_c.score(validation_data,validation_label.ravel()),clf_c.score(test_data,test_label.ravel())])

clf_c_df = pd.DataFrame(s,columns=['C', 'Training', 'Validation','Test'])

print(clf_c_df)


plt.plot(clf_c_df['C'],clf_c_df['Test'])
plt.plot(clf_c_df['C'],clf_c_df['Training'])
plt.plot(clf_c_df['C'],clf_c_df['Validation'])
plt.xlabel("C values")
plt.ylabel("Accuracies")
plt.legend()

############# Full tranining dataset #############

clf = SVC(C = 40,kernel='rbf')
clf.fit(train_data, train_label.ravel())
  
clftrain = clf.score(train_data,train_label.ravel())
clfvalidate = clf.score(validation_data,validation_label.ravel())
clftest = clf.score(test_data,test_label.ravel())
  
print("######################################## Result for SVM with best parameters ########################################")
print("Training set Accuracy : ",str(clftrain))
print("Validation Set Accuracy: ",str(clfvalidate))
print("Test Set Accuracy : ",str(clftest))

	
