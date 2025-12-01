import os
import sys
import logging
import pandas as pd
from config_uploader import ConfigUploader
from azure_handler import AzureBlobHandler
import db_processor
import video_processor

# Configurar el logger principal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(module)-15s | %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
cfg = ConfigUploader()

def main():
    """
    Punto de entrada para el script de subida.
    Llamado por scheduler.py
    Orquesta el flujo de ETL:
    1. Procesar BBDD -> 2. Fusionar Videos -> 3. Subir Videos -> 4. Enriquecer CSV -> 5. Subir CSV
    """
    logger.info("--- Iniciando Proceso ETL Programado ---")

    # --- 0. Conectar a Azure ---
    try:
        azure = AzureBlobHandler(
            conn_str=cfg.AZURE_STORAGE_CONNECTION_STRING,
            container_name=cfg.AZURE_CONTAINER_NAME,
            account_name=cfg.AZURE_ACCOUNT_NAME,
            account_url=cfg.AZURE_STORAGE_ACCOUNT_URL
        )
    except Exception as e:
        logger.critical(f"Fallo al conectar con Azure. Terminando. Error: {e}")
        sys.exit(1)

    # --- 1. Encontrar y Procesar BBDD (Transformación 1) ---
    db_file_path = db_processor.find_db()
    if not db_file_path:
        logger.info("Terminando script (sin BBDD).")
        sys.exit(0)

    summary_df = db_processor.process_risk_events(db_file_path)
    if summary_df.empty:
        logger.info("No hay datos para subir. Borrando BBDD y terminando.")
        try:
            os.remove(db_file_path)
            logger.info(f"Archivo BBDD {db_file_path} borrado.")
        except Exception as e:
            logger.error(f"Error al borrar BBDD: {e}")
        sys.exit(0)
        
    # --- 2. Fusionar Videos (Transformación 2) ---
    summary_df_with_merged_files = video_processor.merge_incident_videos(summary_df)
    
    # --- 3. Subir Videos Finales a Azure (Carga 1) ---
    video_url_map = video_processor.upload_final_videos(summary_df_with_merged_files, azure)

    # --- 4. Enriquecer con Metadatos y URLs (Transformación 3) ---
    enriched_df = db_processor.enrich_summary_data(summary_df_with_merged_files, video_url_map)

    # --- 5. Guardar y Subir CSV Final (Carga 2) ---
    summary_csv_path = os.path.join(cfg.LOG_DIR, f"summary_results.csv")
    summary_db_path = os.path.join(cfg.LOG_DIR, f"summary_results.db")
    try:
        db_processor.write_or_append_csv(enriched_df, summary_csv_path)
        # enriched_df.to_csv(summary_csv_path, index=False, date_format='%Y-%m-%dT%H:%M:%S.%f', encoding='utf-8-sig')
    except Exception as e:
        logger.error(f"Error al guardar el CSV: {e}")
        sys.exit(1)
    try:
        enriched_df
        db_processor.write_or_append_dbsqlite(enriched_df, summary_db_path, "summary_riesgos")
    except Exception as e:
        logger.error(f"Error al guardar la DB: {e}")
        sys.exit(1)

    azure_csv_path = f"{cfg.AZURE_CSV_PATH}/{os.path.basename(summary_csv_path)}"
    if azure.upload_file_and_get_sas_url(summary_csv_path, azure_csv_path):
        # Si la subida fue exitosa, limpiar archivos locales
        logger.info("Limpiando archivos locales post-subida...")
        try:
            # os.remove(db_file_path)
            logger.info(f"Archivo BBDD {db_file_path} borrado.")
            # os.remove(summary_csv_path)
            # logger.info(f"Archivo CSV {summary_csv_path} borrado.")
        except Exception as e:
            logger.error(f"Error durante la limpieza de archivos: {e}")
    else:
        logger.error("La subida del CSV a Azure falló. Los archivos locales se conservarán.")
        sys.exit(1)

    logger.info("--- Proceso ETL Programado Finalizado Exitosamente ---")

if __name__ == "__main__":
    main()