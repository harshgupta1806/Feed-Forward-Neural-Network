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
