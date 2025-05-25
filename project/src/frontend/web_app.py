from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)
# Configuración de la URL base de la API
# Se puede configurar a través de variables de entorno (Docker) o por defecto localhost:8000 (se ejecuta sin Docker)
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/predict_fraud"

# Puerto de la aplicación web
WEB_APP_PORT = os.environ.get("WEB_APP_PORT", "8080")

@app.route("/", methods=["GET"])
def index():
    # Renderiza un formulario HTML
    return render_template("index.html")

@app.route("/predict_fraud", methods=["POST"])
def predict():
    # Recogemos los datos del formulario
    Payment_6804 = request.form.get("Payment_6804")
    Infraction_EJZ = request.form.get("Infraction_EJZ")
    Base_76065 = request.form.get("Base_76065")
    Infraction_GGO = request.form.get("Infraction_GGO")
    Infraction_TLPJ = request.form.get("Infraction_TLPJ")
    Infraction_RKTA = request.form.get("Infraction_RKTA")
    Infraction_CZE = request.form.get("Infraction_CZE")
    Base_02683 = request.form.get("Base_02683")
    Infraction_BSU = request.form.get("Infraction_BSU")
    Infraction_FMXQ = request.form.get("Infraction_FMXQ")
    Infraction_TBP = request.form.get("Infraction_TBP")
    Base_7744 = request.form.get("Base_7744")
    Expenditure_JIG = request.form.get("Expenditure_JIG")
    Risk_8902 = request.form.get("Risk_8902")
    Infraction_AYWV = request.form.get("Infraction_AYWV")

    # Construimos la carga en JSON para la API
    payload = {
        "Payment_6804": float(Payment_6804),
        "Infraction_EJZ": float(Infraction_EJZ),
        "Base_76065": float(Base_76065),
        "Infraction_GGO": float(Infraction_GGO),
        "Infraction_TLPJ": float(Infraction_TLPJ),
        "Infraction_RKTA": float(Infraction_RKTA),
        "Infraction_CZE": float(Infraction_CZE),
        "Base_02683": float(Base_02683),
        "Infraction_BSU": float(Infraction_BSU),
        "Infraction_FMXQ": float(Infraction_FMXQ),
        "Infraction_TBP": float(Infraction_TBP),
        "Expenditure_JIG": float(Expenditure_JIG),
        "Base_7744": float(Base_7744),
        "Risk_8902": float(Risk_8902),
        "Infraction_AYWV": float(Infraction_AYWV)
    }

    # Hacemos la petición POST a la API
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        predicted_price = data["predicted_fraud"]
        return f"La predicción de fraude es: {predicted_price:.2f}"
    else:
        return "Error en la API. No se pudo obtener la predicción."

if __name__ == "__main__":
    # Ejecutar la aplicación web en el puerto especificado
    app.run(host="0.0.0.0", port=WEB_APP_PORT, debug=True)
