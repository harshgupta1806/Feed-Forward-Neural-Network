# Importing necessary libraries
from keras.datasets import fashion_mnist  ## Importing Fashion MNIST dataset from Keras
import numpy as np  # Importing NumPy library for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting
import wandb

#load dataset
(fashion_mnist_train, fashion_mnist_train_label), (fashion_mnist_test, fashion_mnist_test_label) = fashion_mnist.load_data()


label_fahion_mnist = {
     0 :  "T-shirt/top",
     1 :  "Trouser",
     2 :  "Pullover",
     3 :  "Dress",
     4 :  "Coat",
     5 :  "Sandal",
     6 :  "Shirt",
     7 :  "Sneaker",
     8 :  "Bag",
     9 :  "Ankle boot"
}

wandb.init(project="DL_Assignment1", name="Question:1")

# Creating subplots to display one image of each class
fig, axes = plt.subplots(2, 5, figsize=(6, 6))

# Iterating through each class
for i in range(len(label_fahion_mnist)):
    row = i // 5  # Calculating row index for subplot
    col = i % 5   # Calculating column index for subplot

    # Finding the index of the first image in the training set with label i
    idx = np.argmax(fashion_mnist_train_label == i)

    # Displaying the image corresponding to the label i
    axes[row, col].imshow(fashion_mnist_train[idx], cmap='gray')  # Displaying grayscale image
    axes[row, col].set_title(label_fahion_mnist[i])  # Setting title with class label
    axes[row, col].axis('off')  # Turning off axis

    wandb.log({"Question1": [wandb.Image(fashion_mnist_train[idx], caption= label_fahion_mnist[i])]})

# Displaying the plot
plt.show()

