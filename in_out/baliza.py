import socket
import time

# ---Este script de Python realiza una operación simple: 
# ---Conecta a un dispositivo vía TCP/IP, envía un comando para activar un relé, espera 3 segundos, y luego envía otro comando para desactivarlo.

# --- Configuración de la conexión ---
IP_ADDRESS = '192.168.1.101' 
PORT = 2101  

# --- Configuración del temporizador (en segundos) ---
# El relé se mantendrá activado durante este tiempo.
TIMER_SECONDS = 3

# --- Comandos en bytes ---
ACTIVATE_RELAY_1 = bytes([254, 100,1])
DEACTIVATE_RELAY_1 = bytes([254, 101,1])

# --- Lógica del programa ---
print(f"Intentando conectar a {IP_ADDRESS}:{PORT}...")

try:
    # `with` statement se encarga de abrir y cerrar la conexión de forma segura.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((IP_ADDRESS, PORT))
        print("Conexión exitosa.")

        # Paso 1: Activar el relé
        s.sendall(ACTIVATE_RELAY_1)
        print("Comando de activación enviado.")

        # Paso 2: Esperar el tiempo definido en el temporizador
        print(f"Relé activado. Esperando {TIMER_SECONDS} segundos...")
        time.sleep(TIMER_SECONDS)

        # Paso 3: Desactivar el relé
        s.sendall(DEACTIVATE_RELAY_1)
        print("Comando de desactivación enviado.")
        
    print("El relé se ha desactivado. Conexión cerrada.")

except socket.error as e:
    print(f"Error de conexión: {e}")
    print("Asegúrate de que la IP y el puerto sean correctos y el controlador esté encendido.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")