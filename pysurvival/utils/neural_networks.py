import torch 
import torch.nn as nn
import numpy as np
import pysurvival.utils.optimization as opt

# --------------------------- Activation Functions --------------------------- #
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(- x*x/2.)
    
class Atan(nn.Module):
    def forward(self, x):
        return torch.atan(x)

class InverseSqrt(nn.Module):
    def forward(self, x, alpha=1.):
        return x/torch.sqrt(1.+alpha*x*x)
    
class Sinc(nn.Module):
    def forward(self, x, epsilon=1e-9):
        return torch.sin(x+epsilon)/(x+epsilon)
    
class SinReLU(nn.Module):
    def forward(self, x):
        return torch.sin(x)+torch.relu(x)

class CosReLU(nn.Module):
    def forward(self, x):
        return torch.cos(x)+torch.relu(x)

class LeCunTanh(nn.Module):
    def forward(self, x):
        return 1.7159*torch.tanh(2./3*x)
           
class LogLog(nn.Module):
    def forward(self, x):
        return 1.-torch.exp(-torch.exp(x))
    
class BipolarSigmoid(nn.Module):
    def forward(self, x):
        return (1.-torch.exp(-x))/(1.+torch.exp(-x)) 
    
class BentIdentity(nn.Module):
    def forward(self, x, alpha=1.):
        return x + (torch.sqrt(1.+ x*x)- 1.)/2.

class Identity(nn.Module):
    def forward(self, x):
        return x 

class Softmax(nn.Module):
    def forward(self, x):
        y = torch.exp(x)
        return y/torch.sum(y, dim=0)
    
def activation_function(activation, alpha=1., return_text=False):
    """ Returns the activation function object used by the network """
    
    if activation.lower() == 'atan':
        if return_text :
            return 'Atan'
        else:
            return Atan()
    
    elif activation.lower().startswith('bent'):
        if return_text :
            return 'BentIdentity'
        else:
            return BentIdentity()

    elif activation.lower().startswith('bipolar'):
        if return_text :
            return 'BipolarSigmoid'
        else:
            return BipolarSigmoid()
    
    elif activation.lower().startswith('cosrelu'):
        if return_text :
            return 'CosReLU'
        else:
            return CosReLU()    
    
    elif activation.lower() == 'elu':
        if return_text :
            return 'ELU'
        else:
            return nn.ELU(alpha=alpha)
    
    elif activation.lower() == 'gaussian':
        if return_text :
            return 'Gaussian'
        else:
            return Gaussian()
        
    elif activation.lower() == 'hardtanh':
        if return_text :
            return 'Hardtanh'
        else:
            return nn.Hardtanh()
    
    elif activation.lower() == 'identity':
        if return_text :
            return 'Identity'
        else:
            return Identity()
    
    elif activation.lower().startswith('inverse'):
        if return_text :
            return 'InverseSqrt'
        else:
            return InverseSqrt()    
        
    elif activation.lower() == 'leakyrelu':
        if return_text :
            return 'LeakyReLU'
        else:
            return nn.LeakyReLU()
    
    elif activation.lower().startswith('lecun'):
        if return_text :
            return 'LeCunTanh'
        else:
            return LeCunTanh()    
    
    elif activation.lower() == 'loglog':
        if return_text :
            return 'LogLog'
        else:
            return LogLog()
    
    elif activation.lower() == 'logsigmoid':
        if return_text :
            return 'LogSigmoid'
        else:
            return nn.LogSigmoid()    
    
    elif activation.lower() == 'relu':
        if return_text :
            return 'ReLU'
        else:
            return nn.ReLU()
        
    elif activation.lower() == 'selu':
        if return_text :
            return 'SELU'
        else:
            return nn.SELU()
    
    elif activation.lower() == 'sigmoid':
        if return_text :
            return 'Sigmoid'
        else:
            return nn.Sigmoid()
    
    elif activation.lower() == 'sinc':
        if return_text :
            return 'Sinc'
        else:
            return Sinc()
    
    elif activation.lower().startswith('sinrelu'):
        if return_text :
            return 'SinReLU'
        else:
            return SinReLU()    
    
    elif activation.lower() == 'softmax':
        if return_text :
            return 'Softmax'
        else:
            return Softmax()
        
    elif activation.lower() == 'softplus':
        if return_text :
            return 'Softplus'
        else:
            return nn.Softplus()
        
    elif activation.lower() == 'softsign':
        if return_text :
            return 'Softsign'
        else:
            return nn.Softsign()
    
    elif activation.lower() == 'swish':
        if return_text :
            return 'Swish'
        else:
            return Swish()
        
    elif activation.lower() == 'tanh':
        if return_text :
            return 'Tanh'
        else:
            return nn.Tanh()

    else:
        error = "{} function isn't implemented".format(activation)
        raise NotImplementedError(error)



def check_mlp_structure(structure):
    """ Checking that the given MLP structure is valid """


    # Checking if structure is dict
    if isinstance(structure, dict):
        structure = [structure]

    # Checking the keys 
    results = []
    for inner_structure in structure:

        # Checking the validity of activation
        activation = inner_structure.get('activation')
        if activation is None:
            error = 'An activation function needs to be provided ' 
            error +='using the key "activation"'
            raise KeyError(error)

        else:
            activation = activation_function(activation, return_text=True)
            inner_structure['activation'] = activation

        # Checking the validity of num_units
        num_units = inner_structure.get('num_units')
        if num_units is None:
            error = 'The number of hidden units needs to be provided ' 
            error +='using the key "num_units"'
            raise KeyError(error)

        else:
            if not isinstance(num_units, int):
                error = 'num_units in {} needs to be a integer'
                error = error.format(inner_structure)
                raise TypeError(error)
            else:
                inner_structure['num_units'] = num_units

        results.append(inner_structure)

    return results


# ----------------------------- MLP Object ----------------------------- #
class NeuralNet(nn.Module):
    """ Defines a Multilayer Perceptron (MLP) that consists in 
        * an input layer,
        * at least one fully connected neural layer (or hidden layer)
        * and an output layer

    Parameters:
    -----------
    * input_size: int
        Dimension of the input tensor
    * output_size: int
        Size of the output layer
    * structure: None or list of dictionnaries
        Provides the structure of the MLP built within the N-MTLR
        If None, then the model becomes the Linear MTLR
        ex: structure = [ {'activation': 'relu', 'num_units': 128}, 
                          {'activation': 'tanh', 'num_units': 128}, ] 
        Here are the possible activation functions:
            * Atan
            * BentIdentity
            * BipolarSigmoid
            * CosReLU
            * ELU
            * Gaussian
            * Hardtanh
            * Identity
            * InverseSqrt
            * LeakyReLU
            * LeCunTanh
            * LogLog
            * LogSigmoid
            * ReLU
            * SELU
            * Sigmoid
            * Sinc
            * SinReLU
            * Softmax
            * Softplus
            * Softsign
            * Swish
            * Tanh
    * init_method: str
        Defines the type of initializer to use
    * dropout: double (default=None)
        Randomly sets a fraction rate of input units to 0 
        at each update during training time, which helps prevent overfitting.
    * batch_normalization: bool (default=True)
        Applying Batch Normalization or not
    * bn_and_droupout: bool (default=False)
        Applying Batch Normalization and Dropout at the same time


    Note about Dropout and Batch Normalization:
    ------------------------------------------
    As a rule, the dropout Layer and Batch Normalization (BN) shouldn't be used 
    together according to : https://arxiv.org/pdf/1801.05134.pdf

    * Dropout is used to Prevent Neural Networks from Overfitting
      should appears after the activation according to : 
      https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

    * Batch Normalization can Accelerate Deep Network Training by Reducing 
      Internal Covariate Shift BN should appear after Fully connected but 
      before activation according to : https://arxiv.org/pdf/1502.03167.pdf

    """    
    def __init__(self, input_size, output_size, structure, init_method
        , dropout=None, batch_normalization = True, bn_and_droupout = False):

        # Initializing the model
        super(NeuralNet, self).__init__()

        # Initializing the list of layers
        self.layers = []

        if structure is not None and structure != []:

            # Checking if structure is dict
            if isinstance(structure, dict):
                structure = [structure]

            # Building the hidden layers
            for hidden in structure:

                # Extracting the hidden layer parameters 
                hidden_size = int(hidden.get('num_units'))
                activation  = hidden.get('activation')
                alpha       = hidden.get('alpha')

                # Fully connected layer
                fully_conn = nn.Linear(input_size, hidden_size) 
                fully_conn.weight = opt.initialization(init_method, 
                    fully_conn.weight)
                fully_conn.bias = opt.initialization(init_method, 
                    fully_conn.bias)
                self.layers.append( fully_conn )
                
                if not bn_and_droupout:
                    # Batch Normalization
                    if batch_normalization:
                        self.layers.append( torch.nn.BatchNorm1d(hidden_size) )

                    # Activation
                    self.layers.append( activation_function(activation, 
                        alpha=alpha) )
                    
                    # Dropout
                    if (dropout is not None or 0. < dropout <= 1.) and \
                    not batch_normalization :
                        self.layers.append( torch.nn.Dropout(dropout) )

                else:
                    # Batch Normalization
                    if batch_normalization:
                        self.layers.append( torch.nn.BatchNorm1d(hidden_size) )

                    # Activation
                    self.layers.append( activation_function(activation, 
                        alpha=alpha) )
                    
                    # Dropout
                    if (dropout is not None or 0. < dropout <= 1.) :
                        self.layers.append( torch.nn.Dropout(dropout) )

                # Next layer
                input_size = hidden_size

        # Fully connected last layer
        fully_conn = nn.Linear(input_size, output_size) 
        fully_conn.weight = opt.initialization(init_method, fully_conn.weight)
        fully_conn.bias = opt.initialization(init_method, fully_conn.bias)
        self.layers.append( fully_conn )

        # Putting the model together 
        self.model = nn.Sequential(*self.layers).train()


    def forward(self, x):

        out = self.model(x)
        return out


        
class ParametricNet(torch.nn.Module):
    """ Underlying Pytorch model powering the Parametric models """

    def __init__(self, num_features, init_method, init_alpha=1., 
        is_beta_used = True):
        super(ParametricNet, self).__init__()

        # weights
        W = torch.randn(num_features, 1) 
        self.W = opt.initialization(init_method, W)

        one =  torch.FloatTensor(np.array([1]))/init_alpha
        self.alpha = torch.nn.Parameter( one ) 

        self.is_beta_used = is_beta_used
        if self.is_beta_used:
            one =  torch.FloatTensor(np.array([1.001]))/init_alpha
            self.beta = torch.nn.Parameter( one ) 

    def forward(self, x):
        score =  self.alpha*torch.exp(torch.matmul(x, self.W))
        return score

