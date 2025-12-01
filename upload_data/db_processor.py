import os
import sqlite3
import pandas as pd
import json
import sys
import logging
from config_uploader import ConfigUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
cfg = ConfigUploader()

def find_db():
    """Encuentra el archivo .db."""
    db_name = f"registros_riesgos.db"
    db_path = os.path.join(cfg.LOG_DIR, db_name)
    
    if not os.path.exists(db_path):
        logger.warning(f"No se encontró el archivo de log: {db_path}")
        return None
    return db_path

def process_risk_events(db_path):
    """
    Lee la BBDD de SQLite, la procesa con Pandas para agrupar
    eventos de riesgo basados en el tiempo.
    """
    logger.info(f"Procesando archivo: {db_path}")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM riesgos WHERE risk_active = 1", conn)
        
        if df.empty:
            logger.warning("No se encontraron eventos de riesgo activos en la BBDD.")
            return pd.DataFrame() # Devolver DataFrame vacío

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ordenar por escena y luego por tiempo
        df = df.sort_values(by=['scene_name', 'timestamp'])
        # Encontrar los "saltos" de tiempo
        time_diff = df.groupby('scene_name')['timestamp'].diff()
        
        # Identificar el inicio de un nuevo incidente
        is_new_incident = (time_diff.isna()) | (time_diff > pd.Timedelta(seconds=cfg.GAP_THRESHOLD_SECONDS))
        # Crear un ID de incidente único (usando cumsum)
        incident_grouper = is_new_incident.cumsum()
        
        def aggregate_video_files(series):
            unique_files = series.dropna().unique().tolist()
            return unique_files if unique_files else [] # Devolver lista vacía en lugar de None

        # Agrupar por escena y ID de incidente, y agregar
        summary_df = df.groupby(['scene_name', incident_grouper.rename('incident_id')]).agg(
            t_start=('timestamp', 'min'),
            t_end=('timestamp', 'max'),
            # Si se requiere, crear una lista de archivos de video para este incidente
            video_files=('video_file', aggregate_video_files) 
        ).reset_index()

        # Calcular la duración
        summary_df['duration_seconds'] = (summary_df['t_end'] - summary_df['t_start']).dt.total_seconds()
        logger.info(f"Procesamiento de BBDD completo. {len(df)} registros consolidados en {len(summary_df)} incidentes.")
        return summary_df

    except Exception as e:
        logger.error(f"Error al procesar la BBDD con Pandas: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def enrich_summary_data(summary_df, video_url_map):
    """
    Toma el DataFrame de incidentes y lo enriquece con metadatos
    y las URLs de los videos finales.
    """
    logger.info(f"Cargando metadatos desde {cfg.METADATA_FILE_PATH}")
    try:
        with open(cfg.METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
            metadata_json = json.load(f)
        metadata_df = pd.DataFrame.from_dict(metadata_json, orient='index')
        metadata_df = metadata_df.reset_index().rename(columns={'index': 'scene_name'})
        
        logger.info("Uniendo incidentes con metadatos...")
        enriched_df = pd.merge(summary_df, metadata_df, on='scene_name', how='left')
    except Exception as e:
        logger.error(f"Error cargando o uniendo metadatos: {e}. Continuará sin metadatos.")
        enriched_df = summary_df # Continuar sin enriquecimiento
        
    # Enriquecer con URLs de Video
    if not video_url_map:
        logger.warning("No hay mapa de URLs de video. La columna 'video_url' estará vacía.")
        enriched_df['video_url'] = None
    else:
        logger.info("Mapeando URLs de video a los incidentes...")
        enriched_df['video_url'] = enriched_df['final_video_file'].map(video_url_map)
    
    # Limpiar columnas internas
    final_cols = [col for col in enriched_df.columns if col not in ['incident_id', 'video_files', 'final_video_file']]
    enriched_df = enriched_df[final_cols]
    
    return enriched_df

def write_or_append_csv(df, ruta_csv):
    """
    Guarda un DataFrame en un archivo CSV.
    Si el archivo ya existe, agrega las filas sin escribir el encabezado de nuevo.
    """
    # Verifica si el archivo ya existe
    archivo_existe = os.path.isfile(ruta_csv)
    
    # Escribe o agrega según el caso
    df.to_csv(
        ruta_csv,
        mode='a' if archivo_existe else 'w',  # 'a' = append, 'w' = write
        header=not archivo_existe,            # Solo escribe encabezado si el archivo no existe
        date_format='%Y-%m-%dT%H:%M:%S.%f',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ Datos {'agregados' if archivo_existe else 'guardados'} correctamente en {ruta_csv}")

def write_or_append_dbsqlite(df, ruta_db, nombre_tabla):
    """
    Guarda un DataFrame en una base de datos SQLite.
    Si la tabla ya existe, agrega las filas nuevas sin borrar las anteriores.
    """

    # Crear conexión (si no existe la BD, se crea)
    conexion = sqlite3.connect(ruta_db)

    # Insertar datos
    df.to_sql(
        nombre_tabla,
        conexion,
        if_exists='append',  # 'append' agrega los datos, 'replace' sobrescribe
        index=False
    )

    conexion.close()
    print(f"✅ Datos guardados/agregados correctamente en la tabla '{nombre_tabla}' de {ruta_db}")