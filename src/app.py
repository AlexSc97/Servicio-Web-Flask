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
    try:
        # Recoger todos los valores del formulario
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        # Crear el array de numpy en el orden correcto que el modelo espera
        features = np.array([[
            pregnancies, 
            glucose, 
            blood_pressure, 
            skin_thickness, 
            insulin, 
            bmi, 
            dpf, 
            age
        ]])

        # Realizar la predicción
        prediction = model.predict(features)

        # Interpretar el resultado
        if prediction[0] == 1:
            result_text = "Resultado: Positivo para Diabetes"
        else:
            result_text = "Resultado: Negativo para Diabetes"

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        # Si ocurre un error, lo mostramos para facilitar el debug
        error_message = f"Error al procesar la predicción: {e}"
        return render_template('index.html', prediction_text=error_message)


# Permitir que la aplicación se ejecute
if __name__ == '__main__':
    app.run(debug=True)
