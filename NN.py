import numpy as np
from PIL import Image
import os

def preprocess_image(filepath, target_size=(32, 32)):
    image = Image.open(filepath).convert('L')
    image = image.resize(target_size)
    return np.asarray(image, dtype=np.float32).flatten() / 255.0

def read_data(csv_file, img_dir):
    X_data = []
    y_data = []

    with open(csv_file, 'r') as file:
        for row in file:
            img_name, label = row.strip().split(',')
            img_path = os.path.join(img_dir, img_name)
            X_data.append(preprocess_image(img_path))
            y_data.append(float(label))

    return np.array(X_data), np.array(y_data).reshape(-1, 1)

class CompactNet:
    def __init__(self, n_inputs, n_hidden):
        self.weights_input = np.random.randn(n_inputs, n_hidden) * 0.01
        self.bias_input = np.zeros((1, n_hidden))
        self.weights_output = np.random.randn(n_hidden, 1) * 0.01
        self.bias_output = np.zeros((1, 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_grad(self, activated):
        return activated * (1 - activated)

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_grad(self, activated):
        return 1 - activated ** 2

    def run_forward(self, batch):
        self.hidden_linear = np.dot(batch, self.weights_input) + self.bias_input
        self.hidden_activated = self._sigmoid(self.hidden_linear)
        self.output_linear = np.dot(self.hidden_activated, self.weights_output) + self.bias_output
        self.final_output = self._tanh(self.output_linear)
        return self.final_output

    def mse_loss(self, target, prediction):
        return np.mean((target - prediction) ** 2)

    def run_backward(self, batch, target, prediction, lr):
        grad_output = 2 * (prediction - target) / target.shape[0]
        grad_weights_out = np.dot(self.hidden_activated.T, grad_output)
        grad_bias_out = np.sum(grad_output, axis=0, keepdims=True)

        grad_hidden = np.dot(grad_output, self.weights_output.T)
        grad_hidden_linear = grad_hidden * self._tanh_grad(self.hidden_activated)
        grad_weights_in = np.dot(batch.T, grad_hidden_linear)
        grad_bias_in = np.sum(grad_hidden_linear, axis=0, keepdims=True)

        self.weights_output -= lr * grad_weights_out
        self.bias_output -= lr * grad_bias_out
        self.weights_input -= lr * grad_weights_in
        self.bias_input -= lr * grad_bias_in

def train_model(model, features, targets, num_epochs=1000, lr=0.01):
    for epoch in range(num_epochs):
        predictions = model.run_forward(features)
        loss = model.mse_loss(targets, predictions)
        model.run_backward(features, targets, predictions, lr)

        if epoch % 100 == 0:
            print(f"Эпоха {epoch}, Ошибка: {loss:.4f}")

def evaluate(model, img_file):
    sample = preprocess_image(img_file).reshape(1, -1)
    result = model.run_forward(sample)
    print(f"Оценка компактности: {result[0,0]:.3f}")
    return f"{result[0,0]:.3f}"

def launch(test_image_path):
    features, targets = read_data('compic.csv', 'images')
    compact_estimator = CompactNet(n_inputs=1024, n_hidden=64)
    train_model(compact_estimator, features, targets, num_epochs=1000, lr=0.01)
    return evaluate(compact_estimator, test_image_path)
