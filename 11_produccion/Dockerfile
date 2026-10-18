# Imagen CPU. Para GPU NVIDIA: FROM nvcr.io/nvidia/pytorch:<versión>-py3 (o pytorch/pytorch:*-cuda*-runtime)
# y ejecutar con: docker run --gpus all ...
FROM python:3.12-slim

# librerías del sistema que necesita OpenCV para leer vídeo
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
# 1º dependencias (capa cacheada mientras requirements no cambie), 2º código
COPY requirements-app.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt

COPY app/ app/
# Los pesos se montan o se descargan; no se meten en la imagen si cambian a menudo
ENV WEIGHTS=/models/yolo11n.pt LOG_LEVEL=INFO
EXPOSE 8000
# usuario sin privilegios
RUN useradd -m appuser
USER appuser
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Construir y ejecutar:
#   docker build -t vision-api 11_produccion
#   docker run -p 8000:8000 -v $PWD/data:/models vision-api
