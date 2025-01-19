import joblib
import pandas as pd

loaded_model = joblib.load('logistic_regression_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Пример новых данных для предсказания (замените на реальные данные)
new_data = pd.DataFrame({
    'PostsCount': [45],
    'LikesPerPost': [700],
    'MessagesSent': [800],
    'PhotosPerDay': [2.3],
    'Topic': [0]
})

new_data_1 = pd.DataFrame({
    'PostsCount': [200],
    'LikesPerPost': [870],
    'MessagesSent': [1600],
    'PhotosPerDay': [3.0],
    'Topic': [3]
})

# Нормализация новых данных с использованием загруженного нормализатора
new_data_scaled = loaded_scaler.transform(new_data)
new_data_scaled_1 = loaded_scaler.transform(new_data_1)

# Прогнозирование с использованием загруженной модели
new_predictions = loaded_model.predict(new_data_scaled)
new_predictions_1 = loaded_model.predict(new_data_scaled_1)
print(f'Предсказания для новых данных: {new_predictions}')
print(f'Предсказания для новых данных_1: {new_predictions_1}')