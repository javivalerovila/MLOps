FROM python:3.9-slim

RUN apt-get update && apt-get install -y curl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

ENV API_PORT=8000

EXPOSE 8000

# Comando único correcto
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "inference_api:app", "--bind", "0.0.0.0:8000"]