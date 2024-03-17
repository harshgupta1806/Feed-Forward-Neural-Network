from keras.datasets import fashion_mnist  # Importing Fashion MNIST dataset from Keras
import numpy as np  # Importing NumPy library for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import matplotlib
import seaborn as sns
import pandas
import warnings
import argparse

warnings.filterwarnings("ignore")
# wandb.login(key='57566fbb0e091de2e298a4320d872f9a2b200d12')
parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL_Assignment1')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='cs23m026')
parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"])
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=32)
parser.add_argument('-l','--loss', help = 'choices: ["mean_squared_error", "cross_entropy"]' , type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"])
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', type=str, default = 'nadam', choices= ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.0005)
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.9)
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.9)
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.9)
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.999)
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=1e-8)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=0)
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["random", "Xavier"]', type=str, default='Xavier')
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=3)
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', type=int, default=128)
parser.add_argument('-a', '--activation', help='choices: ["sigmoid", "tanh", "ReLU", "identity"]', type=str, default='tanh', choices=["identity", "sigmoid", "tanh", "ReLU"])
parser.add_argument('-p', '--console', help='print training_accuracy + loss, validation_accuracy + loss for every epochs', choices=[0, 1], type=int, default=1)
parser.add_argument('-wl', '--wandb_log', help='log on wandb', choices=[0, 1], type=int, default=1)
parser.add_argument('-cm', '--confusion_matrix', help='log confusion matrix on wandb', choices=[0, 1], type=int, default=0)
# parser.add_argument('-oc', '--output_size', help ='Number of neurons in output layer used in feedforward neural network.', type = int, default = 10)
arguments = parser.parse_args()
# wandb.init(project=  arguments.wandb_project, name = arguments.wandb_entity)

#load dataset

def input_matrix(image):
    """
    Reshape and normalize input images.

    Parameters:
    - image: Input image

    Returns:
    - Reshaped and normalized input matrix
    """
    return image.reshape(image.shape[0], -1) / 255.0



""" Activation Class :- Contains various diffrent activations functions """
class Activation_Functions:
    def __init__(self) -> None:
        pass


    def sigmoid(self, x):
        # Compute sigmoid element-wise for each element of the matrix
        sigmoid_x = np.zeros_like(x)  # Initialize output matrix with zeros

        # Apply the sigmoid function element-wise
        positive = x >= 0
        negetive = ~positive
        sigmoid_x[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        sigmoid_x[negetive] = np.exp(x[negetive]) / (1.0 + np.exp(x[negetive]))

        return sigmoid_x



    def ReLU(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        # Subtract the maximum value along the axis to prevent overflow
        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)

        # Compute softmax probabilities
        softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        return softmax_x

    def identity(self, x):
        return x

    def activation(self, x, fun):
        if fun == "tanh":
            return self.tanh(x)
        elif fun == "sigmoid":
            return self.sigmoid(x)
        elif fun == "ReLU":
            return self.ReLU(x)
        elif fun == "softmax":
            return self.softmax(x)
        elif fun == "identity":
            return self.identity(x)


class Derivatives:
    def __init__(self) -> None:
        """
            Constructor method for Derivatives class.
            Initialization of object of Activation_Functions class
        """
        self.fun = Activation_Functions()  # Creating an instance of Activation_Functions class


    def sigmoid_derivative(self, x):
        """
        Computes the derivative of the sigmoid activation function.

        Parameters:
        - x: Input value

        Returns:
        - Derivative of the sigmoid activation function
        """
        g = self.fun.sigmoid(x)  # Computing sigmoid activation
        return g * (1 - g)  # Computing and returning derivative

    def softmax_derivative(self, x):
        """
        Computes the derivative of the sigmoid activation function.

        Parameters:
        - x: Input value

        Returns:
        - Derivative of the sigmoid activation function
        """
        g = self.fun.softmax(x)  # Computing sigmoid activation
        return g * (1 - g)  # Computing and returning derivative


    def tanh_derivative(self, x):
        """
        Computes the derivative of the hyperbolic tangent (tanh) activation function.

        Parameters:
        - x: Input value

        Returns:
        - Derivative of the tanh activation function
        """
        g = self.fun.tanh(x)  # Computing tanh activation
        return 1 - g * g  # Computing and returning derivative


    def ReLU_derivative(self, x):
        """
        Computes the derivative of the Rectified Linear Unit (ReLU) activation function.

        Parameters:
        - x: Input value

        Returns:
        - Derivative of the ReLU activation function
        """
        # g = self.fun.ReLU(x)  # Computing ReLU activation
        # return np.where(g > 0, 1, 0)  # Computing and returning derivative

        x[x>0]=1
        x[x<=0]=0
        return x

    def identity_derivative(self, x):
        """
        Computes the derivative of the identity activation function.

        Parameters:
        - x: Input value

        Returns:
        - Derivative of the Identity activation function
        """
        x = 1
        return x


    def derivatives(self, x, activation_function):
        """
        Computes the derivative of a specified activation function.

        Parameters:
        - x: Input value
        - activation_function: Name of the activation function

        Returns:
        - Derivative of the specified activation function
        """
        if activation_function == "sigmoid":
            return self.sigmoid_derivative(x)  # Computing derivative for sigmoid activation
        elif activation_function == "tanh":
            return self.tanh_derivative(x)  # Computing derivative for tanh activation
        elif activation_function == "ReLU":
            return self.ReLU_derivative(x)  # Computing derivative for ReLU activation
        elif activation_function == "softmax":
            return self.softmax_derivative(x)
        elif activation_function == "identity":
            return self.identity_derivative(x)


class Loss_Function:
    def __init__(self) -> None:
        """
        Constructor method for Loss_Function class.
        Initializes the default loss function to cross-entropy.
        """
        self.default_loss_function = "cross_entropy"


    def compute_loss(self, y_true, y_hat, loss_function):
        """
        Computes the loss based on the given true labels and predicted probabilities.

        Parameters:
        - y_true: True labels (one-hot encoded)
        - y_hat: Predicted probabilities
        - loss_function: Name of the loss function to be used

        Returns:
        - Loss value
        """


        if loss_function == None:
            # If no loss function is specified, use the default loss function
            loss_function = self.default_loss_function

        if loss_function == "cross_entropy":
            # Set a small value of flag to avoid wrong values in log functions
            flag = 1e-15
            
            y_hat = np.clip(y_hat, flag, 1. - flag) # Clip the predicted values to avoid log(0) and log(1) 
            # Compute the cross-entropy loss for each sample
            loss = -np.sum(y_true * np.log(y_hat), axis=1)
            # Compute the mean loss across all samples
            loss = np.mean(loss)
            # Return the computed loss
            return loss

        if loss_function == "mean_squared_error":
            loss = (1/2) * np.sum((y_true-y_hat)**2) / (y_hat.shape[0])
            return loss


    def last_output_derivative(self, y_hat,y_true, activation_derivative, loss_function):

        if(loss_function == "mean_squared_error"):
            # print(y_hat.shape, y_true.shape)
            return (y_hat - y_true)* activation_derivative/ len(y_true)

        if(loss_function == "cross_entropy"):
            return -(y_true - y_hat)


class Optimizer:
    def __init__(self, neural_network, PARAM) -> None:
        """
        Constructor method for Optimizer class.

        Parameters:
        - neural_network: Instance of the neural network class
        - PARAM: Dictionary containing optimization parameters (eta, weight_decay, optimizer, beta)
        """
        self.neural_network = neural_network  # Neural network object
        self.eta = PARAM["eta"]  # Learning rate
        self.weight_decay = PARAM["weight_decay"]  # Weight decay factor
        self.optimizer = PARAM["optimizer"]  # Optimization algorithm (sgd, momentum, nag)
        self.beta = PARAM["beta"]  # Momentum factor for momentum-based optimization
        self.epsilon = PARAM["epsilon"]
        self.beta2 = PARAM["beta2"]
        self.beta1 = PARAM["beta1"]
        self.momentum = PARAM["momentum"]

    def stochastic_gradient_decent(self):
        """
        Method to perform stochastic gradient descent optimization.
        """
        weight_decay = self.weight_decay
        eta = self.eta

        for layer in range(len(self.neural_network.size_list)-1, 0, -1):
            decay_wt = weight_decay * self.neural_network.weight[layer]  # Applying weight decay
            self.neural_network.grad_w[layer] = self.neural_network.grad_w[layer] + decay_wt  # Adding weight decay to gradients
            self.neural_network.weight[layer] = self.neural_network.weight[layer] - eta * self.neural_network.grad_w[layer]  # Updating weights
            self.neural_network.bias[layer] = self.neural_network.bias[layer] - eta * self.neural_network.grad_b[layer]  # Updating biases

    def update(self, t):
        """
        Method to update network parameters based on the selected optimization algorithm.
        """
        if self.optimizer == "sgd":
            self.stochastic_gradient_decent()
        elif self.optimizer == "momentum":
            self.momentum_based_gradient_decent()
        elif self.optimizer == "nag":
            self.nesterov_accelerated_gradient_decent()
        elif self.optimizer == "rmsprop":
            self.rmsprop()
        elif self.optimizer == "adam":
            self.adam(t)
        elif self.optimizer == "nadam":
            self.nadam(t)


    def momentum_based_gradient_decent(self):
        """
        Method to perform momentum-based gradient descent optimization.
        """
        weight_decay = self.weight_decay

        for layer in range(len(self.neural_network.size_list)-1, 0, -1):
                decay_wt = weight_decay * self.neural_network.weight[layer]  # Applying weight decay
                self.neural_network.grad_w[layer] = self.neural_network.grad_w[layer] + decay_wt  # Adding weight decay to gradients

                uw = self.momentum * self.neural_network.prv_w[layer] + self.eta * self.neural_network.grad_w[layer]  # Computing update for weights
                ub = self.momentum * self.neural_network.prv_b[layer] + self.eta * self.neural_network.grad_b[layer]  # Computing update for biases

                self.neural_network.weight[layer] -= uw  # Updating weights
                self.neural_network.bias[layer] -= ub  # Updating biases

                self.neural_network.prv_w[layer] = uw  # Storing previous weight update
                self.neural_network.prv_b[layer] = ub  # Storing previous bias update


    def nesterov_accelerated_gradient_decent(self):
        """
        Method to perform Nesterov Accelerated Gradient Descent optimization.
        """
        for layer in range(len(self.neural_network.size_list)-1, 0, -1):
            decay_wt = self.weight_decay * self.neural_network.weight[layer]  # Applying weight decay
            self.neural_network.grad_w[layer] = self.neural_network.grad_w[layer] + decay_wt  # Adding weight decay to gradients
            self.neural_network.prv_w[layer] = self.momentum * self.neural_network.prv_w[layer] + self.neural_network.grad_w[layer]
            self.neural_network.prv_b[layer] = self.momentum * self.neural_network.prv_b[layer] + self.neural_network.grad_b[layer]

            self.neural_network.weight[layer] -= ((self.eta) * (self.momentum * self.neural_network.prv_w[layer] + self.neural_network.grad_w[layer]))
            self.neural_network.bias[layer] -= ((self.eta) * (self.momentum * self.neural_network.prv_b[layer] + self.neural_network.grad_b[layer]))

    def rmsprop(self):
        """
        Method to perform Root Mean Square Propogation optimization.
        """
        for layer in range(len(self.neural_network.size_list) - 1, 0, -1):
            decay_wt = self.weight_decay * self.neural_network.weight[layer]  # Applying weight decay
            self.neural_network.grad_w[layer] = self.neural_network.grad_w[layer] + decay_wt  # Adding weight decay to gradients

            self.neural_network.prv_w[layer] = self.beta * self.neural_network.prv_w[layer] + (1 - self.beta) * (self.neural_network.grad_w[layer] ** 2)
            self.neural_network.prv_b[layer] = self.beta * self.neural_network.prv_b[layer] + (1 - self.beta) * (self.neural_network.grad_b[layer] ** 2)

            self.neural_network.weight[layer] -= (self.eta / (np.sqrt(self.neural_network.prv_w[layer] + self.epsilon))) * self.neural_network.grad_w[layer]
            self.neural_network.bias[layer] -= (self.eta / (np.sqrt(self.neural_network.prv_b[layer] + self.epsilon))) * self.neural_network.grad_b[layer]


    def adam(self, t):
        """
        Method to perform adam optimizer.

        Parameter : t (denotes the time stamp in network)
        """
        for layer in range(len(self.neural_network.size_list)-1, 0, -1):
            decay_wt = self.weight_decay * self.neural_network.weight[layer]  # Applying weight decay
            self.neural_network.grad_w[layer] = self.neural_network.grad_w[layer] + decay_wt  # Adding weight decay to gradients

            self.neural_network.prv_w[layer] = self.beta1 * self.neural_network.prv_w[layer] + (1 - self.beta1) * self.neural_network.grad_w[layer]
            self.neural_network.prv_b[layer] = self.beta1 * self.neural_network.prv_b[layer] + (1 - self.beta1) * self.neural_network.grad_b[layer]


            self.neural_network.prv2_w[layer] = self.beta2 * self.neural_network.prv2_w[layer] + (1 - self.beta2) * (self.neural_network.grad_w[layer] ** 2)
            self.neural_network.prv2_b[layer] = self.beta2 * self.neural_network.prv2_b[layer] + (1 - self.beta2) * (self.neural_network.grad_b[layer] ** 2)

            m_w_hat = self.neural_network.prv_w[layer]/(1 - self.beta1**t)
            m_b_hat = self.neural_network.prv_b[layer]/(1 - self.beta1**t)

            v_w_hat = self.neural_network.prv2_w[layer]/(1 - self.beta2 ** t)
            v_b_hat = self.neural_network.prv2_b[layer]/(1 - self.beta2 ** t)

            self.neural_network.weight[layer] -= (self.eta/(np.sqrt(v_w_hat) + self.epsilon)) * m_w_hat
            self.neural_network.bias[layer] -= (self.eta/(np.sqrt(v_b_hat) + self.epsilon)) * m_b_hat


    def nadam(self, t):
        """
        Method to perform nadam optimizer.

        Parameter : t (denotes the time stamp in network)
        """
        for layer in range(len(self.neural_network.size_list)-1, 0, -1):
            decay_wt = self.weight_decay * self.neural_network.weight[layer]  # Applying weight decay
            self.neural_network.grad_w[layer] = self.neural_network.grad_w[layer] + decay_wt  # Adding weight decay to gradients

            self.neural_network.prv_w[layer] = self.beta1 * self.neural_network.prv_w[layer] + (1 - self.beta1) * self.neural_network.grad_w[layer]
            self.neural_network.prv_b[layer] = self.beta1 * self.neural_network.prv_b[layer] + (1 - self.beta1) * self.neural_network.grad_b[layer]

            self.neural_network.prv2_w[layer] = self.beta2 * self.neural_network.prv2_w[layer] + (1 - self.beta2) * (self.neural_network.grad_w[layer] ** 2)
            self.neural_network.prv2_b[layer] = self.beta2 * self.neural_network.prv2_b[layer] + (1 - self.beta2) * (self.neural_network.grad_b[layer] ** 2)

            m_w_hat = self.neural_network.prv_w[layer]/(1 - self.beta1**t)
            m_b_hat = self.neural_network.prv_b[layer]/(1 - self.beta1**t)

            v_w_hat = self.neural_network.prv2_w[layer]/(1 - self.beta2 ** t)
            v_b_hat = self.neural_network.prv2_b[layer]/(1 - self.beta2 ** t)

            self.neural_network.weight[layer] -= (self.eta/(np.sqrt(v_w_hat) + self.epsilon)) * (self.beta * m_w_hat + ((1 - self.beta1) * self.neural_network.grad_w[layer])/(1 - self.beta1 ** t))
            self.neural_network.bias[layer] -= (self.eta/(np.sqrt(v_b_hat) + self.epsilon)) * (self.beta * m_b_hat + ((1 - self.beta1) * self.neural_network.grad_b[layer])/(1 - self.beta1 ** t))

class Neural_Network:
    def __init__(self, PARAM) -> None:
        """
        Constructor method for Neural_Network class.

        Parameters:
        - PARAM: Dictionary containing network parameters (input size, hidden layer sizes, output size,
                 activation function, training input, training output)
        """
        self.weight = {}  # Dictionary to store weights of each layer
        self.bias = {}    # Dictionary to store biases of each layer
        self.a = {}       # Dictionary to store preactivation of each layer
        self.h = {}       # Dictionary to store activations of each layer
        self.grad_w = {}  # Dictionary to store gradients of weights for each layer
        self.grad_b = {}  # Dictionary to store gradients of biases for each layer
        self.prv_w = {}   # Dictionary to store previous gradients of weights for momentum-based optimization
        self.prv_b = {}   # Dictionary to store previous gradients of biases for momentum-based optimization
        self.prv2_w = {}
        self.prv2_b = {}


        self.activation_function = PARAM["activation_function"]  # Activation function for hidden layers
        self.loss_function = PARAM["loss_function"]  # Activation function for hidden layers
        self.initialization = PARAM["init"]
        self.hidden_layers = PARAM["hidden_layers"]
        self.hidden_layer_sizes = PARAM["hidden_layer_sizes"]
        self.dataset = PARAM["dataset"]

        self.act = Activation_Functions()  # Instance of Activation_Functions class
        self.derivative = Derivatives()     # Instance of Derivatives class
        self.loss = Loss_Function()
        self.prv2_w = {}
        self.prv2_b = {}

        if self.dataset == 'fashion_mnist':
            (train_img, train_lbl), (test_img, test_lbl) = fashion_mnist.load_data()
        elif self.dataset == 'mnist':
            (train_img, train_lbl), (test_img, test_lbl) = mnist.load_data()
            
        # Split data in train and validation dataset
        train_image, validation_image, train_label, validation_label = train_test_split(train_img, train_lbl, test_size= 0.1, random_state=41) 
        
        self.input = input_matrix(train_image) # Reshape train_image
        self.y_true = train_label

        self.val_img = input_matrix(validation_image) # Reshape validation_image
        self.val_true = validation_label 

        self.test_img = input_matrix(test_img) # Reshape test_iimg
        self.test_true = test_lbl

        self.size_list = [self.input.shape[1]] + [self.hidden_layer_sizes for _ in range(self.hidden_layers)] + [10]  # Sizes of all layers




    def initialize_parameters(self):
        """
        Method to initialize weights and biases of the network.
        """
        for layer in range(1, len(self.size_list)):

            self.prv_w[layer] = np.zeros((self.size_list[layer-1], self.size_list[layer]))  # Initializing previous weights for momentum-based optimization with zero
            self.prv_b[layer] = np.zeros((1, self.size_list[layer]))  # Initializing previous biases for momentum-based optimization with zero
            self.prv2_w[layer] = np.zeros((self.size_list[layer-1], self.size_list[layer]))  # Initializing previous weights for momentum-based optimization with zero
            self.prv2_b[layer] = np.zeros((1, self.size_list[layer]))  # Initializing previous biases for momentum-based optimization with zero

            for layer in range(1, len(self.size_list)):
                if self.initialization == "random":
                    self.weight[layer] = np.random.randn(self.size_list[layer-1], self.size_list[layer])  # Initializing weights with random values
                    self.bias[layer] = np.random.randn(1, self.size_list[layer])  # Initializing biases with random values
                    # print("Initialize with random")
                elif self.initialization == "Xavier": # initialization when init is Xavier
                    inpt_w = self.size_list[layer-1]
                    opt_w = self.size_list[layer]
                    inpt_b = 1
                    opt_b = self.size_list[layer]

                    variance_w = 6.0/(inpt_w + opt_w)
                    variance_b = 6.0/(inpt_b + opt_b)

                    self.weight[layer] = np.random.randn(inpt_w, opt_w) * np.sqrt(variance_w) # Initializing weights with random values
                    self.bias[layer] = np.random.randn(inpt_b, opt_b) *np.sqrt(variance_b)

                    # print("Initialize with xavier")

    def forward_propagation(self, x):
        """
        Method to perform forward propagation through the network.

        Parameters:
        - x: Input data

        Returns:
        - Output of the final layer (after applying softmax activation)
        """
        self.h[0] = x  # Input layer
        for layer in range(1, len(self.size_list)-1): # from first hidden layer to last hidden layer
            self.a[layer] = np.dot(self.h[layer-1], self.weight[layer]) + self.bias[layer]  # Computing weighted sum of inputs
            self.h[layer] = self.act.activation(self.a[layer], self.activation_function)  # Applying activation function
        self.a[layer+1] = np.dot(self.h[layer], self.weight[layer+1]) + self.bias[layer+1]  # Computing weighted sum for final layer
        self.h[layer+1] = self.act.activation(self.a[layer+1], "softmax")  # Applying softmax activation
        return self.h[layer+1]  # Returning output of final layer

    def backward_propagation(self, input, y_true, y_hat):
        """
        Method to perform backward propagation through the network.

        Parameters:
        - input: Input data
        - y_true: True labels
        - y_hat: Predicted probabilities

        Returns:
        - Gradients of weights and biases
        """

        activation_derivative = self.derivative.derivatives(self.a[len(self.size_list) - 1], "softmax")
        error_wrt_output = self.loss.last_output_derivative(y_hat, y_true, activation_derivative, self.loss_function)

        for layer in range(len(self.size_list)-1, 1, -1):
            self.grad_w[layer] = np.dot(self.h[layer-1].T, error_wrt_output)  # Computing gradients of weights
            self.grad_b[layer] = np.sum(error_wrt_output, axis=0, keepdims=True)  # Computing gradients of biases

            error_wrt_hidden = np.dot(error_wrt_output, self.weight[layer].T)  # Computing error with respect to hidden layer
            error_wrt_output = error_wrt_hidden * self.derivative.derivatives(self.a[layer-1], self.activation_function)  # Computing error with respect to output of hidden layer

        self.grad_w[1] = np.dot(input.T, error_wrt_output)  # Computing gradients of weights for input layer
        self.grad_b[1] = np.sum(error_wrt_output, axis=0, keepdims=True)  # Computing gradients of biases for input layer

        return self.grad_w, self.grad_b  # Returning gradients of weights and biases

    def one_hot_vector(self, y_true):
        """
        Method to convert true labels into one-hot vectors.

        Parameters:
        - y_true: True labels

        Returns:
        - One-hot vector representation of true labels
        """
        vec = np.zeros(10)  # Initializing one-hot vector
        vec[y_true] = 1  # Setting the corresponding index to 1
        return vec

    def one_hot_matrix(self, y_true):
        """
        Method to convert true labels into one-hot matrices.

        Parameters:
        - y_true: True labels

        Returns:
        - One-hot matrix representation of true labels
        """
        row = y_true.shape[0]  # Number of samples
        col = 10  # Number of classes
        mat = np.zeros((row, col))  # Initializing one-hot matrix
        for i in range(row):
            mat[i][y_true[i]] = 1  # Setting the corresponding index to 1
        return mat

class Train_Model:
    def __init__(self, neural_network, optimizer, log, console = 1) -> None:
        """
        Constructor method for Train_Model class.

        Parameters:
        - PARAM_NEURAL_NETWORK: Dictionary containing parameters for the neural network
        - PARAM_OPTIMIZER: Dictionary containing parameters for the optimizer
        """
        self.neural_network =neural_network  # Neural network instance
        self.optimizer = optimizer  # Optimizer instance
        self.loss = Loss_Function()  # Loss function instance
        self.wan_log = log # signifies either to log values or not 
        self.console_log = console # signifies either to print accuracy in console or not for every epoch 

    def compute_performance(self, data, label):
        """
        Computes the performance metrics (loss and accuracy) of the neural network on the given data.

        Parameters:
        - self: Instance of the class containing the method.
        - data: Input data for which performance metrics are to be computed.
        - label: True labels corresponding to the input data.

        Returns:
        - loss: Computed loss value using the specified loss function.
        - accuracy: Accuracy percentage achieved by the neural network on the given data.
        """
        # Perform forward propagation to obtain predictions
        y_predictions = self.neural_network.forward_propagation(data)
        
        # Convert true labels to one-hot encoding for comparison
        labels = self.neural_network.one_hot_matrix(label)
        
        # Compute accuracy by comparing predicted labels with true labels
        accuracy = np.sum(np.argmax(y_predictions, axis=1) == np.argmax(labels, axis=1))
        
        # Compute loss using the specified loss function
        loss = self.loss.compute_loss(labels, y_predictions, self.neural_network.loss_function)
    
        # return both loss and accuracy
        return loss, (accuracy/len(data)) * 100

    def predict_prob(self, data):
        """
        Computes the probabilities of the neural network on the given data.

        Parameters:
        - self: Instance of the class containing the method.
        - data: Input data for which performance metrics are to be computed.

        Returns:
        - probabilities : Predicted probabilities from neural network on given data.
        """
        # Perform forward propagation to obtain predictions
        y_predictions = self.neural_network.forward_propagation(data)

        # return prediction
        return y_predictions

    def fit_data(self, batch_size, epochs):
        """
        Method to train the model on the given dataset.

        Parameters:
        - batch_size: Size of each batch
        - epochs: Number of epochs for training
        """
        # print("fitDataCalled")
        self.neural_network.initialize_parameters()  # Initializing parameters of the neural network
        total_batches = int(np.ceil(self.neural_network.input.shape[0] / batch_size))  # Total number of batches
        for i in range(epochs):
            t = 1
            for batch in range(total_batches):
                batch_start = batch * batch_size  # Starting index of the batch
                batch_end = batch_start + batch_size  # Ending index of the batch
                image_set = self.neural_network.input[batch_start : batch_end]  # Extracting batch of input images
                res_set = self.neural_network.y_true[batch_start : batch_end]  # Extracting batch of true labels


                y_hat = self.neural_network.forward_propagation(image_set)  # Forward propagation
                res = self.neural_network.one_hot_matrix(res_set)  # Converting true labels into one-hot matrices
                grad_w , grad_b = self.neural_network.backward_propagation(image_set, res, y_hat)  # Backward propagation

                for layer in range(1, len(self.neural_network.size_list)):
                    self.neural_network.grad_w[layer] = grad_w[layer]/batch_size  # Normalizing gradients of weights
                    self.neural_network.grad_b[layer] = grad_b[layer]/batch_size  # Normalizing gradients of biases

                self.optimizer.update(t)  # Updating weights and biases using optimizer
                t += 1
            
            # Compute performance on training data
            t_loss, t_acc = self.compute_performance(self.neural_network.input, self.neural_network.y_true)

            # Compute performance on validation data
            v_loss, v_acc = self.compute_performance(self.neural_network.val_img, self.neural_network.val_true)

            # if we want to log values in wandb dashboard
            if self.wan_log == 1:
                wandb.log({
                    'epoch' : i+1,
                    'training-loss' : t_loss,
                    'training-accuracy' : t_acc,
                    'validation-loss' : v_loss,
                    'validation-accuracy' : v_acc,
                })

            # if we want to log loss and accuracy in console for every epoch
            if self.console_log == 1:
                print(f"epoch:{i+1} :: \n Training-loss : {t_loss}, Training-accuracy:{t_acc}%")    # Printing loss and accuracy for each epoch
                print(f"Validation-loss : {v_loss}, Validation-accuracy:{v_acc}%\n\n")    # Printing loss and accuracy for each epoch

        # return training loss, traininig accuracy, validation loss, validation accuracy
        return t_loss, t_acc, v_loss, v_acc


def create_confusion_matrix(y_pred, y_true):
    """
    Creates a confusion matrix based on the predicted and true labels.

    Parameters:
    - y_pred: Predicted labels generated by the model.
    - y_true: True labels corresponding to the input data.

    Returns:
    - mat: Confusion matrix representing the classification results.
    """
    # Initialize an empty confusion matrix with dimensions based on the number of classes
    mat = np.zeros((y_pred.shape[1], y_pred.shape[1]))
    
    # Obtain the predicted class for each sample
    class_pred = np.argmax(y_pred, axis=1)
    
    # Populate the confusion matrix based on true and predicted labels
    for i in range(y_true.shape[0]):
        mat[y_true[i]][class_pred[i]] += 1
    
    # Convert matrix elements to integers for clarity
    mat = mat.astype(int)
    
    # Return the confusion matrix
    return mat


def plot_confusion_matrix(dataset):
    """
    Plots and logs the confusion matrix for the specified dataset.

    Para:
    - dataset: String indicating the dataset name ('mnist' or any other).

    Returns:
    - None
    """
    # Initialize Weights & Biases run
    # wandb.init(project="DL_Assignment1", name="Question:7")
    
    # Define class labels based on dataset
    class_label = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    if dataset == 'mnist':
        class_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    # Create confusion matrix
    mat = create_confusion_matrix(y_pred, model.neural_network.test_true)
    
    # Convert confusion matrix to DataFrame
    df_confusion = pandas.DataFrame(mat, index=class_label, columns=class_label)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(df_confusion, annot=True, fmt='g', linewidths=4, linecolor='white')
    ax.set_xticklabels(class_label, rotation=90)
    ax.set_yticklabels(class_label, rotation=0)
    plt.title('Confusion Matrix', fontsize=8)
    plt.ylabel("Predicted Class")
    plt.xlabel("True Class")
   
    # Log confusion matrix to Weights & Biases
    if arguments.wandb_log == 1:
        wandb.log({"Confusion_Matrix": wandb.Image(plt)})
    plt.show()
    
    # Finish Weights & Biases run
    wandb.finish()

config_param = {
    "hidden_layers": arguments.num_layers,
    "hidden_layer_sizes" : arguments.hidden_size,
    "activation": arguments.activation, # sigmoid, tanh, ReLU
    "loss_function" : arguments.loss, # mean_squared_error, cross_entropy
    "initialization" : arguments.weight_init, #random, Xavier
    "dataset" : arguments.dataset,
    "learning_rate": arguments.learning_rate, #0.0006
    "optimizer": arguments.optimizer, #sgd, momentum, adam, nadam, rmsprop, nag
    "beta": arguments.beta,
    "weight_decay": arguments.weight_decay,
    "epsilon": arguments.epsilon, #1e-8
    "beta2" : arguments.beta2,  # 0.9
    "beta1" : arguments.beta1,
    "momentum" : arguments.momentum, #0.5,
    "epochs" : arguments.epochs,
    "batch_size" : arguments.batch_size
}

if arguments.wandb_log == 1:
    wandb.init(config=config_param, project=arguments.wandb_project, entity=arguments.wandb_entity, name=arguments.wandb_entity) 

# Dictionary of parameters using for object of Neural_Network
PARAM_NEURAL_NETWORK = {
    "hidden_layers": arguments.num_layers,
    "hidden_layer_sizes" : arguments.hidden_size,
    "activation_function": arguments.activation, # sigmoid, tanh, ReLU
    "loss_function" : arguments.loss, # mean_squared_error, cross_entropy
    "init" : arguments.weight_init, #random, Xavier
    "dataset" : arguments.dataset
}

# Dictionary of parameters using for object of Optimizer
PARAM_OPTIMIZER = {
    "eta": arguments.learning_rate, #0.0006
    "optimizer": arguments.optimizer, #sgd, momentum, adam, nadam, rmsprop, nag
    "beta": arguments.beta,
    "weight_decay": arguments.weight_decay,
    "epsilon": arguments.epsilon, #1e-8
    "beta2" : arguments.beta2,  # 0.9
    "beta1" : arguments.beta1,
    "momentum" : arguments.momentum #0.5
}

# neural network object
nn = Neural_Network(PARAM_NEURAL_NETWORK)

# optimizer class
opt = Optimizer(nn, PARAM_OPTIMIZER)

# object of Train_Model class
model = Train_Model(nn, opt, log=arguments.wandb_log, console=arguments.console)
#train_network
training_loss, training_acc, validation_loss, validation_acc = model.fit_data(batch_size=arguments.batch_size, epochs=arguments.epochs)
#print accuracy and loss
print(f"Training Accuracy :- {training_acc}, Training Loss :- {training_loss}\nValidation Accuracy :- {validation_acc}, Validation Loss :- {validation_loss}")

if arguments.confusion_matrix == 1:
    y_pred = model.predict_prob(model.neural_network.test_img)
    # print(model.compute_performance(model.neural_network.test_img, model.neural_network.test_true))
    create_confusion_matrix(y_pred, model.neural_network.test_true)
    plot_confusion_matrix(model.neural_network.dataset)



# compute loss and accuracy on test data and print it
test_loss, test_acc = model.compute_performance(model.neural_network.test_img, model.neural_network.test_true)
print(f"Test Accuracy :: {test_acc}, Test Loss :: {test_loss}")

# finish wandb run
wandb.finish()





