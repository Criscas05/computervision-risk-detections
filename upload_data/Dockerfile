# upload_data/Dockerfile
FROM python:3.10-slim

# Instalar FFmpeg, que es una dependencia de moviepy para procesar y concatenar videos.
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar scripts de Python
COPY . .

CMD ["python", "scheduler.py"]
