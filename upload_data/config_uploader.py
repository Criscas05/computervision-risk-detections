import os
import pytz
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

class ConfigUploader:

    # --- Configuración de Zonas Horarias y Fechas ---
    UPLOAD_TIMEZONE = os.environ.get("TIMEZONE", "America/Bogota")
    BOGOTA_TZ = pytz.timezone(UPLOAD_TIMEZONE)
    TODAY_STR = datetime.now(BOGOTA_TZ).strftime('%Y-%m-%d')
    TODAY_GLOB_VIDEO = datetime.now(BOGOTA_TZ).strftime('%Y%m%d')

    # --- Configuración de Rutas Locales ---
    LOG_DIR = os.environ.get("LOG_DIR", "logs")
    CLIPS_DIR = os.environ.get("CLIPS_DIR", "risk_clips") # Carpeta de salida (mapeada por Docker)
    METADATA_FILE_PATH = os.environ.get("METADATA_FILE_PATH", "/app/config_data/risk_metadata.json")

    # --- Scheduler ---
    HOURS_SCHEDULER_ACTIVE = os.environ.get("HOURS_SCHEDULER_ACTIVE", "[1, 4, 7, 10, 13, 16, 19, 22]")
    MINUTE_SCHEDULER_ACTIVE = int(os.environ.get("MINUTE_SCHEDULER_ACTIVE", "2"))

    # --- Configuración de Lógica de Negocio ---
    GAP_THRESHOLD_SECONDS = int(os.environ.get("GAP_THRESHOLD_SECONDS", "10"))

    # --- Configuración de Azure ---
    AZURE_ACCOUNT_NAME = os.environ.get("AZURE_ACCOUNT_NAME")
    AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME")
    AZURE_CSV_PATH = os.environ.get("AZURE_CSV_PATH", "risk_detection_csv_results")
    AZURE_VIDEO_PATH = os.environ.get("AZURE_VIDEO_PATH", "risk_detection_video_clips_results")
    AZURE_STORAGE_ACCOUNT_URL = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    SAS_EXPIRATION_DAYS = int(os.environ.get("SAS_EXPIRATION_DAYS", "1000"))