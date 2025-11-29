from flask import Flask, render_template, request
import pickle
import numpy as np

import os

import logging
import sys
import xgboost as xgb

# Configurar logging para que se vea en los logs de Render
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta donde se almacena el modelo
# Usamos os.path para obtener la ruta absoluta y evitar errores en Render
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'modelo_xg_optimizado.pkl')

# Cargar el modelo al iniciar la aplicación
model = None
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Modelo cargado exitosamente desde {model_path}")
    else:
        logger.error(f"El archivo del modelo no se encuentra en: {model_path}")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")

# Definir la ruta principal que renderiza el formulario HTML
@app.route('/')
def home():
    return render_template('index.html')

# Definir la ruta para recibir los datos y hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise Exception("El modelo no está cargado. Revisa los logs del servidor.")

        # Recoger todos los valores del formulario
        logger.debug("Recibiendo datos del formulario...")
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
        
        logger.debug(f"Features para predicción: {features}")

        # Realizar la predicción
        prediction = model.predict(features)
        logger.info(f"Predicción realizada: {prediction}")

        # Interpretar el resultado
        if prediction[0] == 1:
            result_text = "Resultado: Positivo para Diabetes"
        else:
            result_text = "Resultado: Negativo para Diabetes"

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        # Si ocurre un error, lo mostramos para facilitar el debug
        logger.error(f"Error en la predicción: {e}", exc_info=True)
        error_message = f"Error al procesar la predicción: {e}"
        return render_template('index.html', prediction_text=error_message)


# Permitir que la aplicación se ejecute
if __name__ == '__main__':
    app.run(debug=True)
