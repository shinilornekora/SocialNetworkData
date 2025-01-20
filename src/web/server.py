from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
import pandas as pd
import os
import json
import numpy as np
from .helper import predictWithKnownWeights

themeType = {
    'political': 0,
    'music': 1,
    'technologies': 2,
    'sport': 3,
    'lifestyle': 4
}

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            script_dir = os.path.dirname(__file__)
            file_path = os.path.join(script_dir, 'index.html')
            try:
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'index.html not found')
        elif self.path == '/handle_click':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Button pressed!')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/handle_click':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                messages = int(data.get('messages', 0))
                photos = float(data.get('photos', 0))
                posts = int(data.get('posts', 0))
                likes = int(data.get('likes', 0))
                topic = data.get('topic', '')
                model = data.get('model', '')

                if (model == 'custom'):
                    # Загрузка нормализационных параметров
                    script_dir = os.path.dirname(__file__)
                    params_file = os.path.join(script_dir, '..\\normalization_params.txt')
                    
                    with open(params_file, 'r') as file:
                        NPRaw = file.readlines()
                        print('ALL ARGS: ', len(NPRaw))
                        
                        NPRaw = [[ float(value) for value in NPline.split(' ') ] for NPline in NPRaw]
                        NPRaw = [np.array(arr) for arr in NPRaw]
                        
                    X_mean = NPRaw[0]
                    X_std = NPRaw[1]

                    print('OK with reading normalization params')

                    # Загрузка весов и нормализационных параметров
                    weights_file = os.path.join(script_dir, '..\\weights.txt')
                    with open(weights_file, 'r') as f:
                        weights = np.array([float(w) for w in f.readlines()])

                    print('OK with reading weight params')

                    # Подготовка данных
                    input_data = np.array([[posts, likes, messages, photos, themeType[topic]]])

                    # Предсказание
                    prediction = predictWithKnownWeights(input_data, weights, X_mean, X_std)

                    print('PREDICTION: ', prediction)
                else:                
                    inputData = pd.DataFrame({
                        'PostsCount': [posts],
                        'LikesPerPost': [likes],
                        'MessagesSent': [messages],
                        'PhotosPerDay': [photos],
                        'Topic': [themeType[topic]]
                    })
                    
                    script_dir = os.path.dirname(__file__)
                    logisticRegressionModel = os.path.join(script_dir, '../logistic_regression_model.pkl')
                    scalerFile = os.path.join(script_dir, '../scaler.pkl')
                    
                    loaded_model = joblib.load(logisticRegressionModel)
                    loaded_scaler = joblib.load(scalerFile)
                    new_data_scaled = loaded_scaler.transform(inputData)
                    
                    prediction = loaded_model.predict(new_data_scaled)
                    
                response = {
                    'message': f'Статус активности пользователя - {prediction}'
                }
            except (ValueError, KeyError) as e:
                print(e)
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Invalid input'}).encode('utf-8'))
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=MyHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Serving on port {port}...')
    httpd.serve_forever()