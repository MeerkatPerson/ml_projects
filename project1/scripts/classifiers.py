import numpy as np
from utils import *
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This module contains the implementation of classifiers objects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classifier superclass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Classifier:
    """ 
    Abstract class to represent a classifier
    """
    def __init__(self):
        """ 
            Sets the parameters and create the parameters dictionary
        """
        raise NotImplementedError("Please Implement this method")

    def update_params(self):
        """
        The object contains a dictionary of the most important paramenters.
        This function should update such dictionary.
        """
        raise NotImplementedError("Please Implement this method")

        
    def train(self, tx_train, y_train):
        """ 
            Learns a w.
            Arguments:
                - tx_train: ndarray matrix of size N*D
                - y_train: ndarray matrix of size D*1
            Hypothesis: tx_train ndd y_train have the same length
        """        
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, tx):
        """ 
            Returns a list of predictions. 
            For linear classifiers with classes {-1,1}, it just returns sign(self.score(x))
            Argument:
                - tx : N*D dimension
            Returns : 
                List[int] of size N
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, predictions, y):
        """
            Computes the accuracy for a list of predictions.
            It counts the number of good predictions and divides it by the number of samples.
            Arguments :
                - predictions : List of size N
                - y : List of size N
            Returns
                float : the accuracy
        """
        return np.sum(predictions==y) / len(y)

    def get_params_and_results(self, tx_train, tx_test, y_train, y_test):
        """
            Returns a dictionnary with the parameters and the accuracy
            example of output : 
            {
                'name' : 'Classifier',
                'accuracy_train' : 0.8, 
                'accuracy_test' : 0.78,
                'params' : 
                {
                    'lambda_' : 0.01,
                    'n_iterations' : 10000,
                    'gamma' = 0.2
                }
            
            }
            Arguments : 
                tx_train : N * D train set
                tx_test : N' * D' test set
                y_train : N * 1 train labels 
                y_test : N' * 1 test labels
            Returns :
                dictionnary of parameters and accuracy
        """
        # Compute my predictions
        predictions_train = self.predict(tx_train)
        predictions_test = self.predict(tx_test)
        
        #construct the final dictionnary
        res = dict()
        res['accuracy_train'] = self.accuracy(predictions_train, y_train)
        res['accuracy_test'] = self.accuracy(predictions_test, y_test)
        res['params'] = self.params
        return res



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Contains all the concrete implementations of classifiers 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##########################################################################################################################

#Linear regression (Contains Ridge Regression and Lasso )
class ClassifierLinearRegression(Classifier):

    def update_params(self):
        """Update the dictionary containng the parameters"""
        self.params['name'] = self.name
        self.params['lambda'] = self.lambda_
        self.params['regularizer'] = self.regularizer

    def __init__(self, lambda_, regularizer):
        """ 
            Sets the parameters
            Argument:
                - lambda_ : float parameter for the ridge regression
                - regularizer: string, defines the type of regularizer used
        """

        #A dictionary containing the relevant parameters of the classifier 
        self.params = dict()

        #name of the classifier 
        self.name = 'ClassifierLinearRegression'

        #kind of regularizer: L1, L2 or None
        self.regularizer = regularizer
        
        #weight of the regularization term in the loss
        self.lambda_= lambda_

        #update the parameters
        self.update_params()

    def train(self, y_train, tx_train):
        """ 
            Trains the model. It learns a w with ridge regression.
            Arguments:
                - tx_train: ndarray matrix of size N*D
                - y_train: ndarray matrix of size D*1
            Hypothesis: tx_train ndd y_train have the same length
        """
        #depending on the regularizer, build different matrices

        #AKA RIDGE REGRESSION
        if self.regularizer == 'L2':       
            aI = self.lambda_ * np.identity(tx_train.shape[1])
        #AKA LASSO
        elif self.regularizer == 'L1':      
            return NotImplementedError('Lasso not implemented yet')
        #AKA Linear regression
        elif self.regularizer == None:
            aI = 0. * np.identity(tx_train.shape[1])
        
        #compute and store the weights

        a = tx_train.T.dot(tx_train) + aI
        b = tx_train.T.dot(y_train)
        self.w =  np.linalg.solve(a, b)     

    def predict(self, tx):
        """ 
            Returns a list of predictions 
            Argument:
                - tx : N*D dimension
            Returns : 
                List[int] of size N
        """
        return np.sign(tx.dot(self.w))


##########################################################################################################################
        
#compute centroids of each class and use it to compute prefiction (the class with closest classifier)
class ClassifierCentroids(Classifier):
    def update_params(self):
        """Update the dictionary containng the parameters"""
        #the name of the classifier
        self.params['name'] = self.name

    def __init__(self):
        """
        Computes the centroids in the training set.
        The prediction is the label of the closer centroid
        """
        #most important parameters of the classifier
        self.params = dict()
        self.name = 'ClassifierCentroids'
        self.update_params()

    def train(self, tx_train, y_train):
        self.centroids = [x[1] for x in build_centroids( y_train, tx_train)]
        self.classes = [x[0] for x in build_centroids( y_train, tx_train)]
    
    def predict(self, tx):

        predictions = np.ones(tx.shape[0]).reshape(1, -1)
        for c in self.centroids:
            p = np.linalg.norm(tx - c, axis = 1).reshape(1, -1)
            predictions = np.append(predictions, p, axis = 0)
        predictions = predictions[1:]
        
        predictions = np.argmin(predictions, axis=0)
        predictions = np.array([ self.classes[p] for p in predictions ])
        return predictions


############################################################################################################################################

#Logistic regression
class ClassifierLogisticRegression(Classifier):

    def update_params(self):
        """Update the dictionary containng the parameters"""
        self.params['name'] = self.name
        self.params['lambda_'] = self.lambda_
        self.params['regulairizer'] = self.regularizer
        self.params['gamma'] = self.gamma
        self.params['max_iterations'] = self.max_iterations
        self.params['threshold'] = self.threshold
    
    def __init__(self, lambda_, regularizer, gamma, max_iterations , threshold):
        """ 
            Sets parameters for logistic regression
            Argument:
                - gamma (float)
                - n_iterations (int)
        """

        #A dictionary containing the relevant parameters of the classifier 
        self.params = dict()

        #name of the classifier 
        self.name = 'LogisticRegression'
        
        #weight of the regularization term in the loss
        self.lambda_= lambda_
        
        #kind of regularizer: L1, L2 or None
        self.regularizer = regularizer

        #the step in gradient descent
        self.gamma = gamma

        #the maximum number of iterations
        self.max_iterations = max_iterations

        #threshold in gradient descent
        self.threshold = threshold
        

        self.update_params()

    def train(self, y_train, tx_train, batch_size = -1, verbose = True, tx_validation = None, y_validation = None, store_gradient = False, store_losses = False):
        """ 
            Trains the model. It learns a new w with logistic regression. 
            Arguments:
                - tx_train: ndarray matrix of size N*D
                - y_train: ndarray matrix of size D*1
            Hypothesis: tx_train ndd y_train have the same length
        """
        #initialize the weights
        self.w = np.zeros(tx_train.shape[1])
        #if store_lossesstore the losses over 1 complete iteration (epoch)
        self.losses = []
        #store the prediction accuracies if validation tests are inputted:
        if (tx_validation is not None) and (y_validation is not None):
            self.pred_accuracies_train = []
            self.pred_accuracies_validation = []
        #store the norm of the gradient if required
        if store_gradient:
            self.stored_gradients = []

        #initiazlie the number of samples
        N = tx_train.shape[0]
        #initialize the batch size
        if batch_size == -1:
            batch_size = N

        #iterate over the dataset
        for iter in range(self.max_iterations):
            
            #loss accumulated over many batches
            acc_loss = 0
            for b in range(0, N, batch_size):  
                
                #perform a gradient step over a batch
                #if required, get also the gradient
                if store_gradient:
                    l, self.w, grad = learning_by_gradient_descent(
                        y_train[b:b+batch_size], 
                        tx_train[b:b+batch_size], 
                        self.w, 
                        self.gamma, 
                        self.lambda_,
                        return_gradient = True)
                    
                else:
                    l, self.w = learning_by_gradient_descent(
                        y_train[b:b+batch_size], 
                        tx_train[b:b+batch_size], 
                        self.w, 
                        self.gamma,
                        self.lambda_
                        )
            
                #update accumulated loss
                acc_loss += l

            #output the loss if verbose
            if verbose and iter % 100 == 0:
                print("Current iteration={a}, loss={b}".format(a=iter, b=acc_loss))
            
            #if required, store the predictions log
            if (tx_validation is not None) and (y_validation is not None):
                self.pred_accuracies_train += [(self.predict(tx_train) == y_train).mean()]
                self.pred_accuracies_validation += [(self.predict(tx_validation) == y_validation).mean()]

            #if required, store the norm of the gradient
            if store_gradient:
                self.stored_gradients += [np.linalg.norm(grad)]

            #store the loss over an iteration
            self.losses += [acc_loss]
            

            #check if convergence has been achieved
            if len(self.losses) > 1 and np.abs(self.losses[-1] - self.losses[-2]) < self.threshold:
            
                print('hit thresh')

                #update internal parameters and exit
                self.params['weights'] = self.w
                
                #if accuracies were required:
                if (tx_validation is not None) and (y_validation is not None):
                    self.params['accuyracues_while_training_train'] = self.pred_accuracies_train
                    self.params['accuyracues_while_training_validation'] = self.pred_accuracies_validation                    

                #if required, store the norm of the gradient
                if store_gradient:
                    self.params['stored_gradients'] = self.stored_gradients

                #if required, store losses
                if store_losses:
                    self.params['losses'] = self.losses                
                break

        #end of training: update internal parameters and exit
        self.params['weights'] = self.w

        #if required, store losses
        if store_losses:
                    self.params['losses'] = self.losses 

        #if accuracies were required:
        if (tx_validation is not None) and (y_validation is not None):
            self.params['accuyracues_while_training_train'] = self.pred_accuracies_train
            self.params['accuyracues_while_training_validation'] = self.pred_accuracies_validation
    
        #if required, store the norm of the gradient
        if store_gradient:
            self.params['stored_gradients'] = self.stored_gradients
    

    def predict(self, x):
        """ 
            returns a list of predictions
            Argument:
                - x: a sample vector 1*D 
            Returns : 
                Array[int] 
        """
        pred = sigmoid(x.dot(self.w)) 
        pred = np.rint(pred)
        return pred