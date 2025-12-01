# upload_data/scheduler.py
import time
import subprocess
import sys
import os
from config_uploader import ConfigUploader
from apscheduler.schedulers.blocking import BlockingScheduler
from pytz import timezone

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
cfg = ConfigUploader()

def run_upload_job():
    """
    Función que se ejecutará a las 7 PM.
    Llama a 'upload_logs.py' como un subproceso.
    """
    logger.info(f"[{time.asctime()}] Iniciando tarea diaria de subida de logs...")
    
    # Usamos sys.executable para garantizar que se usa el mismo intérprete de Python que está ejecutando este script.
    script_path = os.path.join(os.path.dirname(__file__), "upload_logs.py")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Salida del script 'upload_logs.py':\n%s", result.stdout)
        logger.info("--- Tarea de carga finalizada exitosamente ---")
    except subprocess.CalledProcessError as e:
        logger.error("--- 🔴 ¡ERROR! El script de carga falló: ---")
        logger.error("STDOUT:\n%s", e.stdout)
        logger.error("STDERR:\n%s", e.stderr)
        logger.error("-------------------------------------------")
    except Exception as e:
        logger.error(f"🔴 ERROR INESPERADO al ejecutar el subproceso: {e}")

if __name__ == "__main__":
    logger.info("Iniciando servicio de programación (scheduler-service) - Cargue de consolidado diario al Azure Storage...")

    try:
        tz = timezone(cfg.UPLOAD_TIMEZONE)
    except Exception:
       logger.warning(f"Zona horaria '{cfg.UPLOAD_TIMEZONE}' no válida. Usando 'UTC'.")
       tz = timezone("UTC")
        
    try:
        hours_str = cfg.HOURS_SCHEDULER_ACTIVE.strip("[]").replace(" ", "")
        minute_int = cfg.MINUTE_SCHEDULER_ACTIVE
    except Exception as e:
        logger.warning(f"Horarios de activación no válidos: {e}. Usando valores por defecto.")
        hours_str = "1,4,7,10,13,16,19,22"
        minute_int = 2
        
    logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
    # Usamos un 'BlockingScheduler' porque este script, es lo único que se ejecutará en este contenedor.
    scheduler = BlockingScheduler(timezone=tz)
    
    # Programamos la tarea 'run_upload_job' para que se ejecute, todos los días en las HOURS_SCHEDULER_ACTIVE horas
    scheduler.add_job(run_upload_job, 'cron', hour=hours_str, minute=minute_int)

    logger.info(f"🟢 [Scheduler] Servicio de programación iniciado.")
    logger.info(f"    Zona horaria: {tz}")
    logger.info(f"    Tarea programada para las horas: {hours_str} en el minuto {minute_int:02d} (diariamente)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🟡 [Scheduler] Deteniendo servicio...")
        scheduler.shutdown()