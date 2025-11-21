from flask import Flask, render_template, request
import pickle
import numpy as np

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta donde se almacena el modelo
model_path = '../models/modelo_xg_optimizado.pkl'

# Cargar el modelo al iniciar la aplicación
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Definir la ruta principal que renderiza el formulario HTML
@app.route('/')
def home():
    return render_template('index.html')

# Definir la ruta para recibir los datos y hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    glucose = float(request.form['glucose'])
    bmi = float(request.form['bmi'])
    age = int(request.form['age'])
    insulin = float(request.form['insulin'])

    # Crear el array de características para el modelo
    # Asegúrate de que el orden de las características coincida con el que espera tu modelo
    features = np.array([[glucose, bmi, age, insulin]])

    # Realizar la predicción
    prediction = model.predict(features)

    # Interpretar el resultado de la predicción
    # Esto puede variar dependiendo de lo que tu modelo devuelva (ej. 0 o 1)
    if prediction[0] == 1:
        result_text = "El modelo predice: Positivo para Diabetes"
    else:
        result_text = "El modelo predice: Negativo para Diabetes"

    # Renderizar la misma página pero con el resultado de la predicción
    return render_template('index.html', prediction_text=result_text)

# Permitir que la aplicación se ejecute
if __name__ == '__main__':
    app.run(debug=True)
