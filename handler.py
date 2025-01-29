import runpod
import os
import subprocess
import signal
import time
import logging
import sglang as sgl
from sglang.runtime.client import RuntimeClient

# Configuration des logs avec timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Variable globale pour le processus SGLang
sglang_process = None
sglang_client = None

def start_sglang_server():
    """Démarre le serveur SGLang"""
    global sglang_process, sglang_client
    try:
        logger.info("=== DÉMARRAGE DU WORKER ===")
        logger.info(f"Version de SGLang: {sgl.__version__}")
        logger.info(f"Configuration GPU: {os.environ.get('NUM_GPUS', '2')} GPUs")
        logger.info("Vérification de l'espace disque...")
        
        # Vérification de l'espace disque
        df = subprocess.check_output(['df', '-h', '/']).decode()
        logger.info(f"Espace disque disponible:\n{df}")
        
        logger.info("Démarrage du serveur SGLang...")
        start_time = time.time()
        
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "--trust-remote-code",
            "--tp", str(os.environ.get("NUM_GPUS", "2")),
            "--host", "0.0.0.0",
            "--port", "30000"
        ]
        logger.info(f"Commande de lancement: {' '.join(cmd)}")
        
        # Redirection des sorties pour capturer les logs en temps réel
        sglang_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        logger.info("Attente du démarrage du serveur et téléchargement du modèle...")
        
        # Lecture des logs en temps réel
        def log_output(pipe, prefix):
            for line in pipe:
                logger.info(f"{prefix}: {line.strip()}")
                
        from threading import Thread
        Thread(target=log_output, args=(sglang_process.stdout, "SERVER")).start()
        Thread(target=log_output, args=(sglang_process.stderr, "ERROR")).start()
        
        # Attente avec feedback
        total_wait = 120  # augmenté à 120 secondes pour le téléchargement du modèle
        for i in range(total_wait):
            if i % 10 == 0:  # log toutes les 10 secondes
                logger.info(f"Initialisation en cours... {i}/{total_wait}s")
            time.sleep(1)
            
            # Vérification si le processus est toujours en vie
            if sglang_process.poll() is not None:
                raise Exception(f"Le serveur s'est arrêté prématurément avec le code: {sglang_process.returncode}")
        
        # Vérification finale
        if sglang_process.poll() is not None:
            logger.error(f"Le serveur s'est arrêté avec le code: {sglang_process.returncode}")
            raise Exception("Le serveur SGLang s'est arrêté de manière inattendue")
        
        logger.info("Initialisation du client SGLang...")
        sglang_client = RuntimeClient(server_addr="http://localhost:30000")
        
        end_time = time.time()
        logger.info(f"Initialisation terminée en {end_time - start_time:.2f} secondes")
        logger.info("=== WORKER PRÊT ===")
        
    except Exception as e:
        logger.error(f"Erreur critique lors du démarrage: {str(e)}")
        raise

def stop_sglang_server():
    """Arrête proprement le serveur SGLang"""
    global sglang_process
    try:
        if sglang_process:
            logger.info("Arrêt du serveur SGLang...")
            sglang_process.send_signal(signal.SIGTERM)
            sglang_process.wait(timeout=30)
            logger.info("Serveur SGLang arrêté avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt du serveur: {str(e)}")

def handler(event):
    """
    Handler principal pour le worker RunPod
    """
    try:
        logger.info("Nouvelle requête reçue")
        logger.info(f"Paramètres de la requête: {event['input']}")
        
        # Récupération des paramètres de la requête
        prompt = event["input"]["prompt"]
        max_length = event["input"].get("max_length", 1024)
        temperature = event["input"].get("temperature", 0.7)
        
        logger.info("Vérification de l'état du serveur...")
        if sglang_process.poll() is not None:
            logger.error("Le serveur n'est plus en cours d'exécution!")
            return {"error": "Le serveur SGLang n'est plus en cours d'exécution"}
        
        logger.info("Génération du texte avec SGLang...")
        start_time = time.time()
        program = sgl.gen(prompt, max_tokens=max_length, temperature=temperature)
        result = sglang_client.run_single(program)
        end_time = time.time()
        
        logger.info(f"Génération terminée en {end_time - start_time:.2f} secondes")
        return {"generated_text": result.output}
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        return {"error": str(e)}

logger.info("Démarrage de l'application...")

# Démarrage du serveur SGLang au lancement
start_sglang_server()

# Enregistrement du handler de nettoyage
import atexit
atexit.register(stop_sglang_server)

logger.info("Configuration terminée, démarrage du serveur RunPod...")

# Démarrage du serveur RunPod
runpod.serverless.start({"handler": handler}) 