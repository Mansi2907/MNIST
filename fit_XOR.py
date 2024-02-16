'''HYPER PARAMETERS :
SIGMOID :   
Learning Rate = 0.2
Epoch = 500

Hyperbolic Tangent Function : 

Learning Rate = 0.1 
Epoch = 500

'''
import numpy as np
from Neural_Network import *

#  XOR input and target
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# model with sigmoid activation
model_sigmoid = Sequential()
model_sigmoid.add(LinearLayer(2, 2))
model_sigmoid.add(SigmoidLayer())
model_sigmoid.add(LinearLayer(2, 1))
model_sigmoid.add(SigmoidLayer())

# Train the model 
learning_rate_sigmoid = 0.2
epochs_sigmoid = 500

for epoch in range(epochs_sigmoid):
    output = model_sigmoid.forward(X)
    loss = np.mean((output - y) ** 2)
    output_gradient = 2 * (output - y) / len(X)
    model_sigmoid.backward(output_gradient, learning_rate_sigmoid)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")


output_sigmoid = model_sigmoid.forward(X)
predictions_sigmoid = np.round(output_sigmoid)
print("Predictions (Sigmoid):")
print(predictions_sigmoid)

# model with hyperbolic tangent activation
model_tanh = Sequential()
model_tanh.add(LinearLayer(2, 2))
model_tanh.add(TanhLayer())
model_tanh.add(LinearLayer(2, 1))
model_tanh.add(TanhLayer())

# Train the model 
learning_rate_tanh = 0.1
epochs_tanh = 500

for epoch in range(epochs_tanh):
    output = model_tanh.forward(X)
    loss = np.mean((output - y) ** 2)
    output_gradient = 2 * (output - y) / len(X)
    model_tanh.backward(output_gradient, learning_rate_tanh)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")


output_tanh = model_tanh.forward(X)
predictions_tanh = np.round(output_tanh)
print("Predictions (Tanh):")
print(predictions_tanh)
model_sigmoid.save_weights("XOR_solved.w")
