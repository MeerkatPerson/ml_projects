import numpy as np
from utils import *
from preprocessing_helpers import *

############################################
#Classifier superclass
###############################################
# Classifier.py
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



###################
#Concrete classifiers
###################

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


#LR
        

#Use centroids to make predictions
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