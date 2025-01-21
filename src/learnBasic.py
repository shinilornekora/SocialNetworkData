import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Загрузка данных
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'data_with_class_int.csv')
data = pd.read_csv(file_path)

# Разделение на признаки и целевую переменную
X = data[['PostsCount', 'LikesPerPost', 'MessagesSent', 'PhotosPerDay', 'Topic']].values
y = data['classes'].values.reshape(-1, 1)

# Нормализация данных
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Сохранение нормализационных параметров
normParamsFilePath = os.path.join(script_dir, 'normalization_params.txt')
np.savetxt(normParamsFilePath, np.vstack((X_mean, X_std)), delimiter=' ')

# Добавление единичного столбца для смещения (bias)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Разделение на обучающую и тестовую выборку
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(0.7 * X.shape[0])
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# 2. Определение функции стоимости

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = -(1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# 3. Реализация градиентного спуска

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        errors = predictions - y
        gradient = (1 / m) * np.dot(X.T, errors)
        weights -= learning_rate * gradient

        # Сохраняем значение функции стоимости для анализа
        cost = cost_function(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# Инициализация весов и параметров
weights = np.zeros((X_train.shape[1], 1))
learning_rate = 0.01
iterations = 1000

# 4. Обучение модели
weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate, iterations)

# 5. Оценка точности
def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= 0.5).astype(int)

y_pred = predict(X_test, weights)
accuracy = np.mean(y_pred == y_test) * 100
print(f'Точность модели: {accuracy:.2f}%')

with open('weights.txt', 'w') as file:
    weightsFilePath = os.path.join(script_dir, 'weights.txt')
    np.savetxt(weightsFilePath, weights, fmt='%f', delimiter=' ')

# Визуализация функции стоимости
plt.plot(range(iterations), cost_history)
plt.title('График функции стоимости')
plt.xlabel('Итерации')
plt.ylabel('Стоимость')
plt.show()

# Матрица ошибок
confusion_matrix = np.zeros((2, 2), dtype=int)
for true, pred in zip(y_test, y_pred):
    confusion_matrix[true[0], pred[0]] += 1

print("Матрица ошибок:")
print(confusion_matrix)

if X.shape[1] - 1 >= 2:
    plt.figure(figsize=(8, 6))
    x_min, x_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    y_min, y_max = X_test[:, 2].min() - 1, X_test[:, 2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel(),
                 np.mean(X_test[:, 3:], axis=0).reshape(1, -1).repeat(len(xx.ravel()), axis=0)]
    probs = sigmoid(np.dot(grid, weights)).reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=50, cmap='viridis', alpha=0.7)
    plt.scatter(X_test[:, 1], X_test[:, 2], c=y_test.ravel(), edgecolor='k', cmap='coolwarm', alpha=0.7)
    plt.title('Разделяющая гиперплоскость')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.colorbar(label='Вероятность класса 1')
    plt.show()
