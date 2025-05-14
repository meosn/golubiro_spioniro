import numpy as np
from PIL import Image
import os

def load_image(path, size=(32, 32)):
    img = Image.open(path).convert('L')
    img = img.resize(size)
    return np.asarray(img, dtype=np.float32).flatten() / 255.0

def load_dataset(csv_path, image_folder):
    images = []
    labels = []

    with open(csv_path, 'r') as f:
        for line in f:
            filename, label = line.strip().split(',')
            path = os.path.join(image_folder, filename)
            images.append(load_image(path))
            labels.append(float(label))

    return np.array(images), np.array(labels).reshape(-1, 1)

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - x ** 2

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.output = np.tanh(self.Z2)
        return self.output

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred, learning_rate):
        dZ2 = 2 * (y_pred - y_true) / y_true.shape[0]
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.tanh_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def train(model, X, y, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = model.compute_loss(y, y_pred)
        model.backward(X, y, y_pred, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

def predict(model, image_path):
    x = load_image(image_path).reshape(1, -1)
    pred = model.forward(x)
    print(f"Предсказанная компактность: {pred[0,0]:.3f}")
    return f"{pred[0,0]:.3f}"

def start(checking):
    X, y = load_dataset('compic.csv', 'images')
    net = SimpleNeuralNet(input_size=1024, hidden_size=64)
    train(net, X, y, epochs=1000, learning_rate=0.01)
    return predict(net, checking)
