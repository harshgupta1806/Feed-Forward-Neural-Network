PARAM_NEURAL_NETWORK = {
    "hidden_layers": 5,
    "hidden_layer_sizes" : 256,
    "activation_function": "ReLU", # sigmoid, tanh, ReLU
    "loss_function" : "cross_entropy", # mean_squared_error, cross_entropy
    "init" : "Xavier", #random, Xavier
    "dataset" : "fashion_mnist"
}

PARAM_OPTIMIZER = {
    "eta": 0.001, #0.0006
    "optimizer": "nadam", #sgd, momentum, adam, nadam, rmsprop, nag
    "beta": 0.5,
    "weight_decay": 0,
    "epsilon": 1e-6, #1e-8
    "beta2" : 0.999,  # 0.9
    "beta1" : 0.999,
    "momentum" : 0.9 #0.5
}

def create_confusion_matrix(y_pred, y_true):
    mat = np.zeros((y_pred.shape[1], y_pred.shape[1]))
    class_pred = np.argmax(y_pred, axis = 1)
    print(class_pred)
    for i in range(y_true.shape[0]):
        mat[y_true[i]][class_pred[i]] += 1
    mat = mat.astype(int)
    print(type(mat))
    return mat


def plot_confusion_matrix(dataset):
    # wandb.init(project="DL_Assignment1", name="Question:7")
    class_label = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    if dataset == 'mnist':
        class_label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    mat = create_confusion_matrix(y_pred, model.neural_network.test_true)
    df_confusion = pandas.DataFrame(mat, index=class_label, columns=class_label)
    plt.figure(figsize=(10, 10))
    # my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow"])
    ax = sns.heatmap(df_confusion, annot=True, fmt='g',linewidths=4, linecolor='white')
    ax.set_xticklabels(class_label,rotation=90)
    ax.set_yticklabels(class_label,rotation=0)
    plt.title('Confusion Matrix', fontsize=8)
    plt.ylabel("Predicted Class")
    plt.xlabel("True Class")
    # wandb.log({"Confusion_Matrix": wandb.Image(plt)})
    # wandb.finish()



nn = Neural_Network(PARAM_NEURAL_NETWORK)
opt = Optimizer(nn, PARAM_OPTIMIZER)

model = Train_Model(nn, opt, log = 0)
training_loss, training_acc, validation_loss, validation_acc = model.fit_data(batch_size=64, epochs=1)

y_pred = model.predict_prob(model.neural_network.test_img)

plot_confusion_matrix(model.neural_network.dataset)