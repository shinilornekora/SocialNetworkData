import numpy as np
import os


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predictWithKnownWeights(data, weights, mean, std):
    """
    Выполняет предсказание на основе известных весов.

    :param data: Входные данные (numpy array)
    :param weights: Известные веса модели (numpy array)
    :param mean: Средние значения для нормализации
    :param std: Стандартные отклонения для нормализации
    :return: Предсказание (0 или 1)
    """
    # Нормализация данных
    normalized_data = (data - mean) / std
    # Добавление столбца для смещения (bias)
    normalized_data = np.hstack((np.ones((normalized_data.shape[0], 1)), normalized_data))
    # Применение сигмоидной функции
    probabilities = 1 / (1 + np.exp(-np.dot(normalized_data, weights)))
    return (probabilities >= 0.5).astype(int)

