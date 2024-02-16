'''
HYPER PARAMETERS:

MODEL 1: 
Learning rate = 0.001
Batch size = 100
Max eopch = 10

MODEL 2: 
Learning rate = 0.001
Batch size = 256
Max eopch = 20

MODEL 3: 
Learning rate = 0.01
Batch size = 100
Max eopch = 20

'''
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score
import keras.datasets
mnist = keras.datasets.mnist
import matplotlib.pyplot as plt
from Neural_Network import *

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

#  hyperparameters 
hyperparameters = [
    {
        "model_name": "Model 1",
        "layers": [784, 100, 10],
        "activation": "Sigmoid",
        "learning_rate": 0.001,
        "batch_size": 100,
        "max_epochs": 10,
        "early_stopping": 5,
        "weights_filepath": "MNIST_model1"
    },
    {
        "model_name": "Model 2",
        "layers": [784, 200, 100, 10],
        "activation": "Tanh",
        "learning_rate": 0.001,
        "batch_size": 256,
        "max_epochs": 20,
        "early_stopping": 5,
        "weights_filepath": "MNIST_model2"
    },
    {
        "model_name": "Model 3",
        "layers": [784, 50, 50, 10],
        "activation": "Tanh",
        "learning_rate": 0.01,
        "batch_size": 100,
        "max_epochs": 20,
        "early_stopping": 5,
        "weights_filepath": "MNIST_model3"
    }
]

# Training loop
for params in hyperparameters:
    # Print summary of the current model
    print("Model:", params["model_name"])
    print("Layers:", params["layers"])
    print("Activation Function:", params["activation"])
    print("Learning rate:", params["learning_rate"])
    print("Batch size:", params["batch_size"])
    print("Max Epochs:", params["max_epochs"])
    print("Early stopping:", params["early_stopping"])

    
    model = Sequential()

    # Adding layers to the model
    for i in range(len(params["layers"]) - 1):
        if params["activation"] == "Sigmoid":
            layer = SigmoidLayer()
        elif params["activation"] == "Tanh":
            layer = TanhLayer()
        elif params["activation"] == "Softmax":
            layer = SoftmaxLayer()
        elif params["activation"] == "CrossEntropy":
            layer = CrossEntropyLossLayer()

        else:
            raise ValueError("Invalid activation function.")
        model.add(LinearLayer(params["layers"][i], params["layers"][i + 1]))
        model.add(layer)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    no_improvement_count = 0

    for epoch in range(params["max_epochs"]):
        
        for i in range(0, X_train.shape[0], params["batch_size"]):
            
            X_batch = X_train[i: i + params["batch_size"]]
            y_batch = y_train[i: i + params["batch_size"]]

            # Forward pass
            output = model.forward(X_batch)
            loss = model.layers[-1].forward(output)
            
            #  training loss
            train_loss = np.mean(loss)
            train_losses.append(train_loss)
            
            # Backward pass
            gradient = model.layers[-1].backward(output - y_batch, params["learning_rate"])
            model.backward(gradient, params["learning_rate"])

        #  validation loss
        val_output = model.forward(X_val)
        val_loss = np.mean(model.layers[-1].forward(val_output))
        val_losses.append(val_loss)

        # Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= params["early_stopping"]:
                print("Early stopping triggered.")
                break

    
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(params["model_name"])
    plt.show()

    
    test_output = model.forward(X_test)
    test_predictions = np.argmax(test_output, axis=1)
    test_accuracy = accuracy_score(np.argmax(y_test, axis=1), test_predictions)
    print("Test accuracy:", test_accuracy)

    
    model.save_weights(params["weights_filepath"])
