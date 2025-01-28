import runpod
import os
import subprocess
import signal
import time
import logging
from sglang import RuntimeClient, gen

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variable globale pour le processus SGLang
sglang_process = None
sglang_client = None

def start_sglang_server():
    """Démarre le serveur SGLang"""
    global sglang_process, sglang_client
    try:
        logger.info("Démarrage du serveur SGLang...")
        logger.info(f"Nombre de GPUs configuré: {os.environ.get('NUM_GPUS', '2')}")
        
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "--trust-remote-code",
            "--tp", str(os.environ.get("NUM_GPUS", "2"))
        ]
        logger.info(f"Commande de lancement: {' '.join(cmd)}")
        
        # Redirection des sorties pour capturer les logs
        sglang_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info("Attente du démarrage du serveur (30s)...")
        time.sleep(30)
        
        # Vérification des logs du processus
        stdout, stderr = sglang_process.stdout, sglang_process.stderr
        if stdout:
            logger.info("Logs stdout du serveur:")
            for line in stdout:
                logger.info(f"SERVER OUT: {line.strip()}")
        if stderr:
            logger.warning("Logs stderr du serveur:")
            for line in stderr:
                logger.warning(f"SERVER ERR: {line.strip()}")
        
        # Vérification si le processus est toujours en vie
        if sglang_process.poll() is not None:
            logger.error(f"Le serveur s'est arrêté avec le code: {sglang_process.returncode}")
            raise Exception("Le serveur SGLang s'est arrêté de manière inattendue")
        
        logger.info("Initialisation du client SGLang...")
        sglang_client = RuntimeClient("http://localhost:30000")
        logger.info("Client SGLang initialisé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur: {str(e)}")
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
        program = gen(prompt, max_tokens=max_length, temperature=temperature)
        result = sglang_client.run_single(program)
        
        logger.info("Génération terminée avec succès")
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