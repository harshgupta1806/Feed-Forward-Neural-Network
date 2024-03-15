sweep_config = {
    'method' : 'random',
    'metric' : { 'goal' : 'maximize', 'name' : 'Accuracy'},
    'name' : 'sweep1_random_final',

    'parameters' : {
        'epochs' : {'values' : [5, 10]},
        'activation' : {'values' : ['identity', 'tanh', 'sigmoid', 'ReLU']},
        'batch_size' : {'values' : [32, 64, 128]},
        'optimizer' : {'values' : ['momentum','sgd', 'nag', 'nadam', 'adam', 'rmsprop']},
        'weight_decay': {'values': [0, 0.0005]},
        'hidden_layer_sizes' : {'values' : [32, 64, 128]},
        'beta': {'values': [0.5, 0.9]}, #rmsprop
        'beta1' : {'values' : [0.9, 0.999]},
        'beta2': {'values': [0.999]},
        'learning_rate': {'values': [0.0005]},
        'initialization':{'values': ['Xavier', 'random']},
        'hidden_layers': {'values': [3, 4, 5]},
        'loss_function' : {'values' : ['cross_entropy', 'mean_squared_error']},
        'epsilon' : {'values' : [1e-6, 1e-8]},
        'momentum' : {'values' : [0.5, 0.9]},
        'dataset' : {'values' : ['fashion_mnist']}
    }
}

def train():
    var1 = wandb.init(project="DL_Assignment1")
    var2 = var1.config

    # wandb.run.name = 'Optimizer:- ' + var2.optimizer + ' Epoch:- ' + str(var2.epochs) + " Avtivation_Function :- " + var2.activation + " Batch_Size :- " + str(var2.batch_size) + " Initialization :- " + var2.initialization + \
    #                 ' layers:-' + str(len(var2.hidden_layers)) +' decay:-' + str(var2.weight_decay) + ' beta:-' + str(var2.beta) + ' learning_rate:-' + str(var2.learning_rate) + \
    #                 ' beta2 :- ' + str(var2.beta)

    wandb.run.name = f"hl_{var2.hidden_layers}_bs_{var2.batch_size}_e_{var2.epochs}_act_{var2.activation}_eta_{var2.learning_rate}_err_{var2.loss_function}_init_{var2.initialization}_hls_{var2.hidden_layer_sizes}_dataset_{var2.dataset}"

    PARAM_NEURAL_NETWORK = {
        "hidden_layers": var2.hidden_layers,
        "hidden_layer_sizes" : var2.hidden_layer_sizes,
        "activation_function": var2.activation, # sigmoid, tanh, ReLU
        "loss_function" : var2.loss_function, # mean_squared_error, cross_entropy
        "init" : var2.initialization, #random, xavier
        "dataset" : var2.dataset
    }

    PARAM_OPTIMIZER = {
        "eta": var2.learning_rate,
        "optimizer": var2.optimizer, #sgd, momentum, adam, nadam, rmsprop, nag
        "beta": var2.beta,
        "weight_decay": var2.weight_decay,
        "epsilon": var2.epsilon,
        "beta2" : var2.beta2,
        "beta1" : var2.beta1,
        "momentum" : var2.momentum
    }

    neural_network1 = Neural_Network(PARAM_NEURAL_NETWORK)
    optimizer1 = Optimizer(neural_network1, PARAM_OPTIMIZER)

    my_model1 = Train_Model(neural_network1, optimizer1, log = 1)
    t_loss, t_acc, v_loss, v_acc = my_model1.fit_data(batch_size=var2.batch_size, epochs=var2.epochs)

    print(f"Training-loss : {t_loss}, Training-accuracy:{t_acc}%, Validation-loss : {v_loss}, Validation-accuracy:{v_acc}%")    # Printing loss and accuracy for each epoch
    loss, accuracy = my_model1.compute_performance(neural_network1.val_img, neural_network1.val_true)
    print(loss, accuracy)
    wandb.log({"Accuracy" : accuracy})

sweep_id = wandb.sweep(sweep_config, project="DL_Assignment1")

wandb.agent(sweep_id, train, count = 250)
wandb.finish()
