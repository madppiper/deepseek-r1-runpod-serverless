FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
RUN pip install -r requirements.txt

# Copie des fichiers du worker
COPY handler.py .

# Variable d'environnement pour le nombre de GPUs
ENV NUM_GPUS=2

# Démarrage du worker RunPod avec SGLang
CMD [ "python", "-u", "handler.py" ] 
