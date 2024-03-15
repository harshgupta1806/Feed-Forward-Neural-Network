---
# Fundamentals of Deep Learning (CS6910) - Assignment 1

This repository hosts a Python script for training a Feed Forward Neural Network implemented from scratch using NumPy. The neural network is highly adaptable, designed to support diverse configurations and easily adjustable for classification tasks, including datasets like MNIST or Fashion MNIST. It offers flexibility for incorporating activation functions, loss functions, and other parameters, facilitating straightforward modifications to suit specific requirements.

---

# Code Structure

## Class `Activation_Functions`

### Attributes:

- `None`

### Methods:

- `sigmoid`: return sigmoid funtion
- `ReLU`: return ReLU function
- `tanh`: return tanh function.
- `softmax`: return softmax function
- `identity`: return identity function
- `activation`: return function based on activation function name
  
## Class `Derivatives`

### Attributes:

- `fun`: Instance of Activation Class

### Methods:

- `sigmoid_derivative`: return derivative of sigmoid funtion
- `ReLU_derivative`: return derivative of ReLU function
- `tanh_derivative`: return derivative of tanh function.
- `softmax_derivative`: return derivative of softmax function
- `identity_derivative`: return derivative of identity function
- `derivatives`: return derivative of function based on activation function

## Class `Loss_Functions`

### Attributes:

- `default_loss_function`: used **cross_entopy** as defult error function

### Methods:

- `compute_loss`
    - `parameters` : `y_true`, `y_hat`, `loss_function`
    - `return` : return the loss based on loss_function
- `last_output_derivative`:
    - `parameter` : `y_true`, `y_hat`, `loss_function`, 'activation_derivative'
    - `return` : return loss based on last layer activation function


## Class `Optimizer`

### Attributes:

- `neural_network`: Reference to the neural network object.
- `Optimizer-specific parameters': `momentum`, `beta`, `beta1`, `beta2`, `epsilon`, `weight_decay`, `optimizer`.

### Methods:

- `Optimization algorithms`: `stochastic_gradient_decent()`, `momentum_based_gradient_decent()`, `nesterov_accelerated_gradient_decent()`, `rmsprop()`, `adam(t)`, `nadam(t)`.
- `update`: Update weights and biases using the selected optimizer.

### Class `Neural_Network`

### Attributes:

- `weight` : Dictionary to store weights of each layer
- `bias` :  Dictionary to store bias of each layer
- `a` :  Dictionary to store pre-activation of each layer
- `h` :  Dictionary to store activation of each layer
- `prv_w` :  Dictionary to store history of gradients wrt to weight of each layer
- `prv_b` :  Dictionary to store history of gradients wrt to bias of  each layer
- `grad_w` :  Dictionary to store gradients wrt to weight of each layer
- `grad_b` :  Dictionary to store gradients wrt to bias of each layer
- `Neural_Network-specific parameters': `activation_function`, `loss_function`, `initialization`, `hidden_layers`, `hidden_layer_sizes`, `dataset`
- `Objects` : `act : Activation_Functions`, `derivative : Derivatives`, `loss : Loss_Function`
- `images` : `train_img`, `y_true`, `val_img`, `val_true`, `test_img` , `test_lbl`

### Methods:

- `initialize_parameters`: Randomly initialize weights and biases, gradients wrt to weights and biases for each layer with proper shapes
- `forward_propagation` : Method to perform forward propagation through the network.
    - `Parameters` :
        - `x`: Input data
    - `Returns` :
        - Output of the final layer (after applying softmax activation)
- `backward_propagation` : Method to perform backward propagation through the network
    - `Parameters` :
        - `input`: Input data
        - `y_true`: True labels
        - `y_hat`: Predicted probabilities
    - `Returns` :
        - Gradients of weights and biases

- `one_hot_matrix`: Method to convert true labels into one-hot matrices.
    - `Parameters` :
        - `y_true`: True labels
    - `Returns` :
        - One-hot vector representation of true labels
- `one_hot_vector`: Method to convert true labels into one-hot vector.
    - `Parameters` :
        - `y_true`: True labels
    - `Returns` :
        - One-hot matrix representation of true labels.

### Class `Train_Model`

### Attributes:

- `weight` : Dictionary to store weights of each layer
- `bias` :  Dictionary to store bias of each layer
- `a` :  Dictionary to store pre-activation of each layer
- `h` :  Dictionary to store activation of each layer
- `prv_w` :  Dictionary to store history of gradients wrt to weight of each layer
- `prv_b` :  Dictionary to store history of gradients wrt to bias of  each layer
- `grad_w` :  Dictionary to store gradients wrt to weight of each layer
- `grad_b` :  Dictionary to store gradients wrt to bias of each layer
- `Neural_Network-specific parameters': `activation_function`, `loss_function`, `initialization`, `hidden_layers`, `hidden_layer_sizes`, `dataset`
- `Objects`
  - `neural_network` : Neural_Network
  - `optimizer` : Optimizer,
  -  `loss` : Loss_Function
- `wan_log`
  - wan_log = 1 => log the values in wandb dashboard
  - wan_log = 0 => not to log values in wandb dashboard
    
### Methods:

- `compute_performance`: Method to compute loss and accuracy
  - `Parameters` :
        - `data`: input data
        - `label`: true_class
    - `Returns` :
        - loss
        - accuracy (in %)
- `predict_prob`: Method to compute loss and accuracy
  - `Parameters` :
        - `data`: input data
    - `Returns` :
        - predicted probabilities assigned to each class
- `fit_data` : Method to train the model on the given dataset.
    - `Parameters` :
        - `batch_size`: Size of each batch
        - `epochs`: Number of epochs for training
    - `Returns` :
        - Training Loss
        - Training Accuracy
        - Validation Loss
        - Validation Accuracy

  
