from flask import Flask, render_template, request, jsonify
import requests
import os
import logging
from flask_wtf.csrf import CSRFProtect

# Configuración inicial
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-123')  # Mejor seguridad para producción
csrf = CSRFProtect(app)  # Protección CSRF

# Configuración logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración API (mejor manejo de errores)
API_CONFIG = {
    'host': os.environ.get("API_HOST", "localhost"),
    'port': os.environ.get("API_PORT", "8000"),
    'endpoint': '/predict_fraud'
}
API_BASE_URL = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}{API_CONFIG['endpoint']}"

# Puerto de la aplicación web
WEB_APP_PORT = int(os.environ.get("WEB_APP_PORT", 8080))

# Lista de campos requeridos (centralizado para validación)
REQUIRED_FIELDS = [
    'Payment_6804', 'Infraction_EJZ', 'Base_76065', 'Infraction_GGO',
    'Infraction_TLPJ', 'Infraction_RKTA', 'Infraction_CZE', 'Base_02683',
    'Infraction_BSU', 'Infraction_FMXQ', 'Infraction_TBP', 'Expenditure_JIG',
    'Base_7744', 'Risk_8902', 'Infraction_AYWV'
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict_fraud", methods=["POST"])
def predict():
    try:
        # Validación de campos
        errors = []
        payload = {}

        for field in REQUIRED_FIELDS:
            value = request.form.get(field)
            if not value:
                errors.append(f"Campo requerido faltante: {field}")
                continue
            try:
                payload[field] = float(value)
            except ValueError:
                errors.append(f"Valor inválido para {field}: {value}")

        if errors:
            return render_template("error.html", errors=errors), 400

        # Llamada a la API
        response = requests.post(API_BASE_URL, json=payload, timeout=10)  # Timeout añadido

        response.raise_for_status()  # Lanza excepción para códigos 4xx/5xx

        data = response.json()
        prediction = data.get('predicted_fraud')

        if prediction is None:
            raise ValueError("Respuesta de API inválida")

        return render_template("result.html",
                            prediction=f"{prediction:.2f}",
                            raw_data=payload)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error en la conexión con la API: {str(e)}")
        return render_template("error.html",
                            errors=["Error al conectar con el servicio de predicción"]), 500

    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        return render_template("error.html",
                            errors=["Ocurrió un error interno en el servidor"]), 500

if __name__ == "__main__":
    # Configuración de entorno segura
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    app.run(host="0.0.0.0",
            port=WEB_APP_PORT,
            debug=debug_mode,
            use_reloader=debug_mode)