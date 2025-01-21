import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

script_dir = os.path.dirname(__file__)

# Загрузка данных из CSV
dataFile = os.path.join(script_dir, 'data_with_class_int.csv')
data = pd.read_csv(dataFile)

X = data[['PostsCount', 'LikesPerPost', 'MessagesSent', 'PhotosPerDay', 'Topic']]
y = data['classes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Инициализация и обучение модели логистической регрессии
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# Функция для создания парных графиков
def plot_pairwise_features(X, y, model):
    feature_names = X.columns
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15), sharex=False, sharey=False)

    for i, feature_x in enumerate(feature_names):
        for j, feature_y in enumerate(feature_names):
            ax = axes[i, j]

            if i == j:
                # Диагональ: гистограммы признаков
                ax.hist(X[feature_x], bins=15, color='gray', alpha=0.7)
                ax.set_title(feature_x)
            else:
                # Вне диагонали: scatter plot с вероятностью классов
                x_min, x_max = X[feature_x].min(), X[feature_x].max()
                y_min, y_max = X[feature_y].min(), X[feature_y].max()
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                     np.linspace(y_min, y_max, 100))

                # Подготовка сетки с фиксацией остальных признаков
                grid = np.c_[xx.ravel(), yy.ravel()]
                other_features = np.mean(X.drop([feature_x, feature_y], axis=1).values, axis=0)
                other_features_repeated = np.tile(other_features, (grid.shape[0], 1))  # Повторение для всех точек
                grid_full = np.column_stack([grid, other_features_repeated])  # Объединение

                # Предсказание вероятностей для сетки
                probs = model.predict_proba(grid_full)[:, 1].reshape(xx.shape)

                # График вероятностей
                ax.contourf(xx, yy, probs, alpha=0.8, levels=50, cmap='viridis')
                scatter = ax.scatter(X[feature_x], X[feature_y], c=y, edgecolor='k', cmap='coolwarm', alpha=0.7)

            # Настройки осей
            if i == n_features - 1:
                ax.set_xlabel(feature_x)
            if j == 0:
                ax.set_ylabel(feature_y)

    plt.tight_layout()
    plt.show()


# Построение графиков
# plot_pairwise_features(X, y, model)


# Создание сетки для двух признаков (feature_1 и feature_2)
x_min, x_max = X_scaled[:, 0].min(), X_scaled[:, 0].max()
y_min, y_max = X_scaled[:, 1].min(), X_scaled[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Зафиксируем остальные признаки на средних значениях
fixed_features = np.mean(X_scaled[:, 2:], axis=0)  # Средние значения для feature_3, feature_4, feature_5

# Создание полной сетки данных
grid = np.c_[xx.ravel(), yy.ravel()]  # Сетка для feature_1 и feature_2
fixed_features_repeated = np.tile(fixed_features, (grid.shape[0], 1))  # Повторение фиксированных признаков
grid_full = np.column_stack([grid, fixed_features_repeated])  # Объединение сетки и фиксированных признаков

# Предсказание вероятностей для каждой точки сетки
probs = model.predict_proba(grid_full)[:, 1].reshape(xx.shape)

# Построение графика
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.predict_proba(X_scaled)[:, 1], cmap='viridis', edgecolor='k')
plt.colorbar(label='Predicted Class (0: Normal, 1: Critical)')
plt.contourf(xx, yy, probs, alpha=0.7, levels=50, cmap='viridis')
plt.title('Logistic Regression Classification (Feature 1 vs Feature 2)')
plt.xlabel('Normalized Feature 1')
plt.ylabel('Normalized Feature 2')
plt.show()

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]  # Вероятности класса 1

# Фиксация признаков на средних значениях
fixed_features = np.mean(X_train[:, 2:], axis=0)  # Средние значения для 3-го, 4-го и 5-го признаков
X_grid = np.c_[np.linspace(0, 1, 100), np.linspace(0, 1, 100)]  # Сетка для двух признаков
grid_full = np.column_stack([X_grid, np.tile(fixed_features, (X_grid.shape[0], 1))])

# Предсказание вероятностей для сетки
grid_probs = model.predict_proba(grid_full)[:, 1]

# Визуализация
print("Plotting classification results...")
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_prob, cmap='coolwarm', s=30, label='Predicted Class')  # Два признака
plt.colorbar(label='Predicted Class (0: Normal, 1: Critical)')
plt.xlabel('Feature 1 (e.g., Temperature)')
plt.ylabel('Feature 2 (e.g., Pressure)')
plt.title('Logistic Regression Classification')
plt.legend()
plt.savefig('classification_results_balanced.png')
plt.show()



# Вычисление точности
accuracy = accuracy_score(y_test, y_pred) * 100
print(f'Точность модели: {accuracy:.2f}%')

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.title('Матрица ошибок')
plt.show()


joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')