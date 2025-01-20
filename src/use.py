import joblib
import os
import pandas as pd
from web.server import run

def exampleflow():
    script_dir = os.path.dirname(__file__)
    logisticRegressionModel = os.path.join(script_dir, 'logistic_regression_model.pkl')
    scalerFile = os.path.join(script_dir, 'scaler.pkl')
    
    loaded_model = joblib.load(logisticRegressionModel)
    loaded_scaler = joblib.load(scalerFile)

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

    new_data_scaled = loaded_scaler.transform(new_data)
    new_data_scaled_1 = loaded_scaler.transform(new_data_1)

    negativePredictionsPositive = loaded_model.predict(new_data_scaled)
    activePredictionsPositive = loaded_model.predict(new_data_scaled_1)

    print(f'Предсказания для новых данных (скорее неактив): {negativePredictionsPositive}')
    print(f'Предсказания для новых данных (скорее актив): {activePredictionsPositive}')
    
def main_executor():
    run()
    
    
# main_executor()
exampleflow()