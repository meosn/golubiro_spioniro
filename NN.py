import numpy as np              # Для работы с числами и матрицами
from PIL import Image           # Для загрузки и обработки изображений
import os                       # Для работы с файлами

# ---------- Загрузка изображения ----------
def load_image(path, size=(32, 32)):
    img = Image.open(path).convert('L')         # Преобразуем в ЧБ (градации серого)
    img = img.resize(size)                      # Приводим к фиксированному размеру (32x32)
    return np.asarray(img, dtype=np.float32).flatten() / 255.0  # Нормализация и преобразуем в 1D

# ---------- Загрузка данных ----------
def load_dataset(csv_path, image_folder):
    images = []  # Список для хранения изображений
    labels = []  # Список для хранения меток (компактности)

    with open(csv_path, 'r') as f:
        for line in f:
            filename, label = line.strip().split(',')  # Читаем имя файла и метку компактности
            path = os.path.join(image_folder, filename)  # Строим путь к изображению
            images.append(load_image(path))  # Загружаем изображение
            labels.append(float(label))  # Сохраняем метку компактности

    return np.array(images), np.array(labels).reshape(-1, 1)  # Возвращаем массивы изображений и меток

# ---------- Простая нейросеть ----------
class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size):
        # Инициализация весов и смещений для двух слоев
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Веса для первого слоя
        self.b1 = np.zeros((1, hidden_size))  # Смещения для первого слоя
        self.W2 = np.random.randn(hidden_size, 1) * 0.01  # Веса для второго слоя
        self.b2 = np.zeros((1, 1))  # Смещения для второго слоя

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Сигмоида

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Производная от сигмоида

    def tanh(self, x):
        return np.tanh(x)  # Функция активации tanh

    def tanh_derivative(self, x):
        return 1 - x ** 2  # Производная от tanh

    def forward(self, X):
        # Прямой проход через нейросеть (Feedforward)
        self.Z1 = np.dot(X, self.W1) + self.b1  # Линейная комбинация для первого слоя
        self.A1 = self.sigmoid(self.Z1)  # Активация первого слоя (сигмоида)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Линейная комбинация для второго слоя
        self.output = np.tanh(self.Z2)  # Активация второго слоя (tanh для ограничений в [-1, 1])
        return self.output

    def compute_loss(self, y_true, y_pred):
        # Функция для вычисления ошибки (среднеквадратичная ошибка)
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred, learning_rate):
        # Обратное распространение ошибки (Backpropagation)
        dZ2 = 2 * (y_pred - y_true) / y_true.shape[0]  # Градиент для второго слоя
        dW2 = np.dot(self.A1.T, dZ2)  # Градиент для весов второго слоя
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # Градиент для смещений второго слоя

        dA1 = np.dot(dZ2, self.W2.T)  # Градиент для активаций первого слоя
        dZ1 = dA1 * self.tanh_derivative(self.A1)  # Градиент для входа первого слоя с учетом производной от tanh
        dW1 = np.dot(X.T, dZ1)  # Градиент для весов первого слоя
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # Градиент для смещений первого слоя

        # Обновление параметров модели
        self.W2 -= learning_rate * dW2  # Обновление весов второго слоя
        self.b2 -= learning_rate * db2  # Обновление смещений второго слоя
        self.W1 -= learning_rate * dW1  # Обновление весов первого слоя
        self.b1 -= learning_rate * db1  # Обновление смещений первого слоя

# ---------- Обучение ----------
def train(model, X, y, epochs=1000, learning_rate=0.01):
    for epoch in range(epochs):
        y_pred = model.forward(X)  # Прямой проход (вычисление предсказания)
        loss = model.compute_loss(y, y_pred)  # Вычисление ошибки
        model.backward(X, y, y_pred, learning_rate)  # Обратное распространение ошибки

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")  # Вывод ошибки каждые 100 эпох

# ---------- Предсказание ----------
def predict(model, image_path):
    x = load_image(image_path).reshape(1, -1)  # Загружаем и преобразуем изображение
    pred = model.forward(x)  # Прогоняем через модель
    print(f"Предсказанная компактность: {pred[0,0]:.3f}")  # Выводим результат
    return f"{pred[0,0]:.3f}"

# ---------- Точка входа ----------
def start(checking):
    # Загружаем данные и тренируем модель
    X, y = load_dataset('compic.csv', 'images')  # Загрузка данных
    net = SimpleNeuralNet(input_size=1024, hidden_size=64)  # Инициализация модели
    train(net, X, y, epochs=1000, learning_rate=0.01)  # Обучение модели

    # Проверка на новом изображении
    return predict(net, checking)  # Прогноз на тестовом изображении
