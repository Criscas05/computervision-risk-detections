# risk_detection/Dockerfile
# Usamos una imagen base oficial de PyTorch con CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Configuración del entorno
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalación de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia del código de la aplicación
COPY . .

# Comando de inicio
# Ejecuta directamente el script de Python. El script maneja su propio ciclo de vida.
CMD ["python", "main_realtime.py"]

