FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Actualizamos paquetes básicos y herramientas de compilación mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copiamos requirements e instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código de la aplicación
COPY app ./app

# Creamos directorios estándar para logs (el modelo lo trae HF/Optimum)
RUN mkdir -p /logs

# Variables de entorno genéricas (se sobreescriben en docker-compose / Actions)
ENV MODEL_ID="" \
    MODEL_TASK="text-classification" \
    MODEL_CLASS="" \
    TOKENIZER_ID="" \
    ENVIRONMENT="dev" \
    LOG_DIR="/logs"

# Exponemos el puerto de la API
EXPOSE 8000

# Comando por defecto: levantar FastAPI con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
