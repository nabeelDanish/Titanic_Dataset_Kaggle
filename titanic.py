#***********************************
#----STEP 1: IMPORTING LIBRARIES
#***********************************
# For data analysis and wrangling
import pandas as pd 
import numpy as np 
import random as rnd 

# For visualization
import seaborn as sns
import matplotlib.pyplot as plt 
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib','inline')

# For machine learning
import tensorflow as tf
from tensorflow.python.framework import ops

# Utility
import math
import time
import h5py
import csv
import pickle
from tf_utils import random_mini_batches, convert_to_one_hot

print("********************************************************")
print("----------[TITANIC SURVIVOR PREDICTION MODEL]-----------")
print("********************************************************")

#***********************************
#----STEP 2: DATASET LOADING
#***********************************
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
combine = [train_df, test_df]

#***********************************
#----STEP 3: DATASET PROCESSING
#***********************************
# Dropping Cabin and Ticket
# these feautures dont contribute to survival

train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
combine = [train_df, test_df]

# Making the Title Feature
# easy to use than names 
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Catgerozing and Correcting Titles
# categorizing the titles into 5 categories 
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Title Mapping
# replacing the strings with ids, giving NULL to 0
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Dropping Name and Passenger ID
# not droping pid from test becuase we need to use it to submit 
# kaggle
train_df = train_df.drop(['Name','PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)

combine = [train_df, test_df]

# Converting Sex to Gender
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Filling the missing age data
# As age is continous, we fill it by finding mean and std of the particular class and sex
guess_ages = np.zeros((2, 3))   #shape = (2, 3) because 2 genders and 3 Pclasses

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1 )]['Age'].dropna()

            age_guess = guess_df.median()
            #Convert age float to nearest 0.5
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

    #Filling the guessed ages where the age is null
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1),\
                        'Age'] = guess_ages[i, j]

    #changing the data type to int
    dataset['Age'] = dataset['Age'].astype(int)

# Creating Age groups
# 5 groups of age created
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# Use this to display age groups
# print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by = 'AgeBand', ascending = True))

# Age Mapping based on grouping
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64), 'Age'] = 4

# Removing the AgeBand column
train_df = train_df.drop(['AgeBand'], axis = 1)
combine = [train_df, test_df]

# Creating feature FamilySize from SibSp and Parch
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Creating feature isAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Dropping Parch and SibSp
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
combine = [train_df, test_df]

# Adding feature Age*Pclass
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# Filling Embarked values
# As embarked is categorical, we fill it from the most occuring
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Mapping Embarked 
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# Filling in the missing Fare values
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)

# Since Fare is also like Age, we need to created bands
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# Mapping the Fare and dropping Fareband
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis = 1)
combine = [train_df, test_df]

# Preparing Datasets for Logistics Regression
X_train_orig = train_df.drop(['Survived'], axis = 1)
X_test_orig = test_df.drop(['PassengerId'], axis = 1).copy()

X_train = np.array(X_train_orig[:][:]).T
X_test = np.array(X_test_orig[:][:]).T

numClasses = 2
Y_train = np.array(train_df['Survived'][:])
Y_train = Y_train.reshape(1, Y_train.shape[0])
Y_train = convert_to_one_hot(Y_train, numClasses)

# Printing the Shapes
print("Dataset Loaded.")
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))

#***********************************
#----STEP 4: INITIALIZING FUNCTIONS
#***********************************
'''The placeholder creater for X and Y as tensorflow placeholders'''
def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, (n_x, None), name = "X")
    Y = tf.placeholder(tf.float32, (n_y, None), name = "Y")
    return X, Y

def initialize_parameters(layer_dims):

    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = tf.get_variable("W" + str(l), [layer_dims[l], layer_dims[l - 1]], initializer = tf.contrib.layers.xavier_initializer())
        parameters["b" + str(l)] = tf.get_variable("b" + str(l), [layer_dims[l], 1], initializer = tf.zeros_initializer())

    return parameters

#***********************************
#STEP 5: FORWARD PROPAGATION
#***********************************    
'''Forward Propagation for a single layer'''
def linear_forward(A, W, b):
    Z = tf.add(tf.matmul(W, A), b)
    return Z

'''Forward Propagation for a single layer with activation'''
def linear_activation_forward(A_prev, W, b):
    Z = linear_forward(A_prev, W, b)
    A = tf.nn.relu(Z)
    return A

'''Forward Propagation for the whole model'''
def forward_propagation(X, parameters):
    A = X
    L = len(parameters) // 2

    #Iteration over the layers
    for l in range(1, L):
        A_prev = A
        A = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])

    #Last Layer Z
    ZL = linear_forward(A, parameters["W"+ str(L)], parameters["b" + str(L)])  

    return ZL  
   
#***********************************
#STEP 6: COMPUTE COST
#***********************************     
def compute_cost(ZL, Y, parameters, beta):

    #For tensorflow requirement
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    L = len(parameters) // 2

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    loss = regularizer(parameters)
    cost = tf.reduce_mean(cost + (loss * beta))

    return cost

#***********************************
#STEP 7: REGULARIZATION
#*********************************** 
def regularizer(parameters):
    L = len(parameters) // 2

    #Regularization
    regular = tf.nn.l2_loss(parameters["W1"])
    for l in range(2, L + 1):
        regular += tf.nn.l2_loss(parameters["W" + str(l)])

    return regular  

#***********************************
#STEP 8: THE MAIN MODEL FUNCTION
#***********************************     
def model(X_train, Y_train, X_test, layer_dims, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):

    #Variables
    ops.reset_default_graph()
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    lambd = 25
    beta = (lambd/m)

    #Creating placeholders
    X, Y = create_placeholders(n_x, n_y)

    #Initialize parameters
    parameters = initialize_parameters(layer_dims)

    #Forward Propagation
    ZL = forward_propagation(X, parameters)

    #Cost
    cost = compute_cost(ZL, Y, parameters, beta)

    #Backpropagation and Optimizing
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    #Tensorflow init
    init = tf.global_variables_initializer()

    #Starting the Session and the Model
    with tf.Session() as sess:

        #Running the init
        sess.run(init)

        #Training Loop
        for epoch in range(num_epochs):

            #Making the Minibatches
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            #Iterating over the minibatches
            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                #Running the session for the whole model
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})

                epoch_cost += minibatch_cost / minibatch_size

            #Printing the cost after 10 epochs
            if print_cost and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)
        #Training Ends
        print ("Cost after epoch %i: %f" % (num_epochs, costs[-1]))
        # Plotting the Cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        
        return parameters


#***********************************
#STEP 10: THE PREDICT FUNCTION
#*********************************** 
def predict_new(X, parameters):

    #Converting the parameters to tensors
    params = {}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        params.update({"W" + str(l): tf.convert_to_tensor(parameters["W" + str(l)])})
        params.update({"b" + str(l): tf.convert_to_tensor(parameters["b" + str(l)])})

    x = tf.placeholder("float", [X.shape[0], X.shape[1]])

    zl = forward_propagation(x, params)
    p = tf.argmax(zl)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    return prediction

#***********************************
#STEP 11: RUNNING THE FINAL MODEL
#*********************************** 
layer_dims = [X_train.shape[0], 650, 450, 375, 290, 180, 100, 70, 20, 2]

print("-------------------------------------------")
print(".............TRAINING THE MODEL............")
print("-------------------------------------------")

parameters = {}

if(input("Do you want to Train the Model? y/n") == "y"):

    #Training
    start_time = time.time()

    parameters = model(X_train, Y_train, X_test, layer_dims, learning_rate = 0.0001)

    #Storing
    print("Storing Learned Parameters")
    with open("params", 'wb+') as f:
        pickle.dump(parameters, f) 
    print("Parameters stored")   

    elapsed_time = time.time() - start_time
    print("Time Taken: " + str(elapsed_time / 60.) + " minutes.")
else:
    print("Loading Parameters from File")
    with open("params", 'rb+') as f:
        parameters = pickle.load(f)
    print("Parameters loaded")

#Making Predictions and storing the result
Y_pred = predict_new(X_test, parameters)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': Y_pred})
submission.to_csv('submission.csv', index=False)