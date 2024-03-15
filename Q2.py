class Network:
    def __init__(self, PARAM):
        """
        Constructor method for Network class.

        Parameters:
        - PARAM: Dictionary containing network parameters (input size, hidden layers, output size)
        """
        self.weight = {}  # Dictionary to store weights of each layer
        self.bias = {}    # Dictionary to store biases of each layer
        self.a = {}       # Dictionary to store activations of each layer
        self.h = {}       # Dictionary to store outputs of each layer
        self.fun = Activation_Functions()  # Instance of Activation_Functions class
        self.size_list = [PARAM["input_size"]] + PARAM["hidden_layers"] + [PARAM["output_size"]]  # List containing sizes of all layers
        self.y_predictions = []  # List to store predicted probabilities for each input sample


    def initialize_parameters(self, initialization):
        """
        Method to initialize weights and biases of the network.
        """
        for layer in range(1, len(self.size_list)):
            if initialization == "random":
                self.weight[layer] = np.random.randn(self.size_list[layer-1], self.size_list[layer])  # Initializing weights with random values
                self.bias[layer] = np.random.randn(1, self.size_list[layer])  # Initializing biases with random values
            elif initialization == "Xavier":
                inpt_w = self.size_list[layer-1]
                opt_w = self.size_list[layer]
                inpt_b = 1
                opt_b = self.size_list[layer]

                variance_w = 2.0/(inpt_w + opt_w)
                variance_b = 2.0/(inpt_b + opt_b)

                self.weight[layer] = np.random.randn(inpt_w, opt_w) * np.sqrt(variance_w) # Initializing weights with random values
                self.bias[layer] = np.random.randn(inpt_b, opt_b) *np.sqrt(variance_b)  # Initializing biases with random values





    def forward_pass(self, x, activation_function):
        """
        Method to perform forward pass through the network.

        Parameters:
        - x: Input data
        - activation_function: Name of the activation function to be used

        Returns:
        - Output of the final layer (after applying softmax activation)
        """
        total_layer = len(self.size_list)
        self.h[0] = x  # Input layer

        for layer in range(1, total_layer-1):
            self.a[layer] = np.dot(self.h[layer-1], self.weight[layer]) + self.bias[layer]  # Computing preactivation
            self.h[layer] = self.fun.activation(self.a[layer], fun=activation_function)  # Applying activation function

        self.a[total_layer-1] = np.dot(self.h[layer], self.weight[total_layer-1]) + self.bias[total_layer-1]  # Computing weighted sum for final layer
        self.h[total_layer-1] = self.fun.activation(self.a[total_layer-1], fun="softmax")  # Applying softmax activation

        return self.h[total_layer-1]  # Returning output of final layer


    def predict_probability(self, dataset, activation, init):
        """
        Method to predict probabilities for each input sample in the dataset.

        Parameters:
        - dataset: Input dataset
        - activation: Name of the activation function to be used

        Returns:
        - List containing predicted probabilities for each input sample
        """
        self.initialize_parameters(init)  # Initializing network parameters

        for image in dataset:
            x = image.reshape(1, -1) / 255.0  # Reshaping and normalizing input data
            y_hat = self.forward_pass(x, activation)  # Performing forward pass
            self.y_predictions.append(y_hat)  # Storing predicted probabilities

        return self.y_predictions  # Returning list of predicted probabilities

PARAM = {
    "input_size" : 784,            # Size of the input layer (number of input features)
    "hidden_layers" : [5, 6, 7],   # Sizes of hidden layers in the neural network
    "output_size" : 10             # Size of the output layer (number of classes)
}

n1 = Network(PARAM)  # Creating an instance of the Network class with the given parameters
y_pred = n1.predict_probability(fashion_mnist_train, "identity", "Xavier")  # Predicting probabilities for each sample in the training dataset using sigmoid activation function

np.set_printoptions(suppress=True)  # Suppressing scientific notation in printed arrays
print(y_pred[0])  # Printing the predicted probabilities for the first sample