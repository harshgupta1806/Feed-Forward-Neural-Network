# Helper Function
def train_model_on_mnist(PARAM_NEURAL_NETWORK, PARAM_OPTIMIZER, batch_size, epochs):
    neural_network_mnist = Neural_Network(PARAM_NEURAL_NETWORK)
    optimizer_mnist = Optimizer(neural_network_mnist, PARAM_OPTIMIZER)

    my_model_mnist = Train_Model(neural_network_mnist, optimizer_mnist, log = 0, console = 0)
    my_model_mnist.fit_data(batch_size, epochs)
    loss, accuracy = my_model_mnist.compute_performance(my_model_mnist.neural_network.test_img, my_model_mnist.neural_network.test_true)
    print(f"Test Data Loss : {loss}, Test Data Accuracy : {accuracy}")

# Model 1
MODEL1_PARAM_NEURAL_NETWORK = {
    "hidden_layers": 3,
    "hidden_layer_sizes" : 32,
    "activation_function": "ReLU", # sigmoid, tanh, ReLU
    "dataset" : "mnist",
    "loss_function" : "cross_entropy", # mean_squared_error, cross_entropy
    "init" : "Xavier" #random, Xavier
}

MODEL1_PARAM_OPTIMIZER = {
    "eta": 0.005,
    "optimizer": "sgd", #sgd, momentum, adam, nadam, rmsprop, nag
    "beta": 0.9,
    "weight_decay": 1e-8,
    "epsilon": 1e-6,
    "beta2" : 0.999,
    "beta1" : 0.999,
    "momentum" : 0.9
}

# Model 2
MODEL2_PARAM_NEURAL_NETWORK = {
    "hidden_layers": 3,
    "hidden_layer_sizes" : 32,
    "activation_function": "ReLU", # sigmoid, tanh, ReLU
    "dataset" : "mnist",
    "loss_function" : "cross_entropy", # mean_squared_error, cross_entropy
    "init" : "Xavier" #random, Xavier
}

MODEL2_PARAM_OPTIMIZER = {
    "eta": 0.001,
    "optimizer": "momentum", #sgd, momentum, adam, nadam, rmsprop, nag
    "beta": 0.9,
    "weight_decay": 0,
    "epsilon": 1e-8,
    "beta2" : 0.999,
    "beta1" : 0.999,
    "momentum" : 0.9
}

# Model 3
MODEL3_PARAM_NEURAL_NETWORK = {
    "hidden_layers": 4,
    "hidden_layer_sizes" : 64,
    "activation_function": "ReLU", # sigmoid, tanh, ReLU
    "dataset" : "mnist",
    "loss_function" : "cross_entropy", # mean_squared_error, cross_entropy
    "init" : "Xavier" #random, Xavier
}

MODEL3_PARAM_OPTIMIZER = {
    "eta": 0.001,
    "optimizer": "rmsprop", #sgd, momentum, adam, nadam, rmsprop, nag
    "beta": 0.9,
    "weight_decay": 0,
    "epsilon": 1e-8,
    "beta2" : 0.999,
    "beta1" : 0.999,
    "momentum" : 0.9
}

# Train Models

train_model_on_mnist(MODEL1_PARAM_NEURAL_NETWORK, MODEL1_PARAM_OPTIMIZER, 32, 1)
train_model_on_mnist(MODEL2_PARAM_NEURAL_NETWORK, MODEL2_PARAM_OPTIMIZER, 32, 1)
train_model_on_mnist(MODEL3_PARAM_NEURAL_NETWORK, MODEL3_PARAM_OPTIMIZER, 32, 1)