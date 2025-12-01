import os
import sys
import logging
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from config_uploader import ConfigUploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
cfg = ConfigUploader()

class AzureBlobHandler:
    """
    Maneja toda la conectividad y subidas a Azure Blob Storage.
    """
    def __init__(self, conn_str, container_name, account_name, account_url):
        if not all([conn_str, container_name, account_name, account_url]):
            logger.error("Faltan una o más credenciales de Azure (ConnectionString, ContainerName, AccountName, AccountURL).")
            raise ValueError("Faltan credenciales de Azure")
            
        self.conn_str = conn_str
        self.container_name = container_name
        self.account_name = account_name
        self.account_url = account_url.rstrip('/') # Asegurar que no tenga / al final
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            logger.info("✅ Conexión con Azure Blob Storage establecida.")
        except Exception as e:
            logger.error(f"Error al conectar con Azure (revisar connection string): {e}")
            raise

    def upload_file_and_get_sas_url(self, local_file_path, azure_file_path):
        """
        Sube un archivo local a Azure y devuelve una URL SAS de lectura.
        Devuelve None si falla.
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=azure_file_path
            )
            logger.info(f"  Subiendo {local_file_path} a {azure_file_path}...")
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"  ✅ Subida completada. Generando URL SAS...")

            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=azure_file_path,
                account_key=self.blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(cfg.BOGOTA_TZ) + timedelta(days=cfg.SAS_EXPIRATION_DAYS)
            )
            
            file_url = f"{self.account_url}/{self.container_name}/{azure_file_path}?{sas_token}"
            logger.info(f"  ✅ URL SAS generada (válida por {cfg.SAS_EXPIRATION_DAYS} días).")
            return file_url

        except Exception as e:
            logger.error(f"  ❌ Fallo al subir {local_file_path}: {e}", exc_info=True)
            return None