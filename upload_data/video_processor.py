import os
import logging
import sys
from moviepy import VideoFileClip, concatenate_videoclips
from config_uploader import ConfigUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
cfg = ConfigUploader()

def merge_incident_videos(summary_df):
    """
    Itera sobre el DataFrame de incidentes, concatena los videos
    que tienen más de un archivo, y actualiza el DataFrame.
    """
    logger.info("Iniciando fase de concatenación de videos...")
    
    summary_df['final_video_file'] = None
    files_to_delete = set()

    for idx, row in summary_df.iterrows():
        video_files = row['video_files']
        
        if not video_files: # 0 videos
            logger.info(f"  Incidente {row['incident_id']} no tiene video.")
            continue
            
        elif len(video_files) == 1: # 1 video
            logger.info(f"  Incidente {row['incident_id']} tiene 1 video, no necesita merge.")
            summary_df.at[idx, 'final_video_file'] = video_files[0]
            
        else: # 2 o más videos (¡necesita merge!)
            logger.warning(f"  Incidente {row['incident_id']} tiene {len(video_files)} videos. Iniciando merge...")
            
            clips_to_merge = []
            valid_files_to_merge = []
            for file_name in video_files:
                file_path = os.path.join(cfg.CLIPS_DIR, file_name)
                if os.path.exists(file_path):
                    clips_to_merge.append(VideoFileClip(file_path))
                    valid_files_to_merge.append(file_path)
                    files_to_delete.add(file_path) # Marcar para borrar
                else:
                    logger.error(f"    No se encontró el archivo {file_path} para el merge.")
            
            if not clips_to_merge:
                logger.error(f"  Incidente {row['incident_id']} no tiene archivos válidos para merge.")
                continue

            base_name = os.path.splitext(video_files[0])[0]
            merged_file_name = f"{base_name}_MERGED.mp4"
            merged_file_path = os.path.join(cfg.CLIPS_DIR, merged_file_name)

            try:
                final_clip = concatenate_videoclips(clips_to_merge)
                final_clip.write_videofile(merged_file_path, codec="libx264", audio=False, logger=None)
                
                summary_df.at[idx, 'final_video_file'] = merged_file_name
                logger.info(f"  ✅ Merge completado → {merged_file_name}")

            except Exception as e:
                logger.error(f"  ❌ Fallo el merge para el incidente {row['incident_id']}: {e}")
                summary_df.at[idx, 'final_video_file'] = video_files[0]
            finally:
                for clip in clips_to_merge:
                    clip.close()

    # Borrar los archivos originales que se fusionaron
    logger.info(f"Borrando {len(files_to_delete)} videos originales post-merge...")
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"  No se pudo borrar el clip original {file_path}: {e}")

    return summary_df

def upload_final_videos(summary_df, azure_handler):
    """
    Sube solo los videos finales (originales o fusionados) a Azure.
    """
    logger.info("Iniciando subida de videos finales a Azure...")
    video_url_map = {} # { "file_name.mp4": "https://...sas" }
    
    final_video_files = summary_df['final_video_file'].dropna().unique()

    if len(final_video_files) == 0:
        logger.info("No hay videos finales para subir.")
        return video_url_map

    logger.info(f"Se encontraron {len(final_video_files)} videos finales para subir...")
    
    for file_name in final_video_files:
        local_file_path = os.path.join(cfg.CLIPS_DIR, file_name)
        if not os.path.exists(local_file_path):
            logger.warning(f"  No se encontró el archivo final {local_file_path} para subir.")
            continue
            
        azure_video_name = f"{cfg.AZURE_VIDEO_PATH}/{file_name}"
        file_url = azure_handler.upload_file_and_get_sas_url(local_file_path, azure_video_name)
        
        if file_url:
            video_url_map[file_name] = file_url
            # Borrar video local si la subida fue exitosa
            try:
                os.remove(local_file_path)
                logger.info(f"  Archivo de video local borrado: {local_file_path}")
            except Exception as e:
                logger.error(f"  Error borrando video local {local_file_path}: {e}")
        else:
            logger.error(f"  No se borrará {local_file_path} debido a fallo en la subida.")
            
    logger.info(f"Subida de videos finales completada: {len(video_url_map)}/{len(final_video_files)} exitosos.")
    return video_url_map