sweep_config_que8 = {
    'method' : 'grid',
    'metric' : { 'goal' : 'maximize', 'name' : 'Accuracy'},
    'name' : 'sweep1_crossEntropy_vs_mse_final',

    'parameters' : {
        'epochs' : {'values' : [10]},
        'activation' : {'values' : ['tanh', 'ReLU']},
        'batch_size' : {'values' : [64]},
        'optimizer' : {'values' : ['momentum', 'nag', 'nadam', 'adam', 'rmsprop', 'sgd']},
        'weight_decay': {'values': [0]},
        'hidden_layer_sizes' : {'values' : [32, 64]},
        'beta': {'values': [0.9]}, #rmsprop
        'beta1' : {'values' : [0.999]},
        'beta2': {'values': [0.999]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'initialization':{'values': ['Xavier']},
        'hidden_layers': {'values': [4, 5]},
        'loss_function' : {'values' : ['cross_entropy', 'mean_squared_error']},
        'epsilon' : {'values' : [1e-6]},
        'momentum' : {'values' : [0.9]},
        'dataset' : {'values' : ['fashion_mnist']}
    }
}


sweep_id_que8 = wandb.sweep(sweep_config_que8, project="DL_Assignment1")
wandb.agent(sweep_id_que8, train)
wandb.finish()