# Étape 1 : Partir d'une image NVIDIA (Ubuntu 22.04)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Mettre à jour l'OS et installer les dépendances
# (Nous gardons cmake et python3.11-dev au cas où une autre dépendance en aurait besoin)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    libpq-dev \
    gcc \
    cmake \
    python3.11 \
    python3-pip \
    python3.11-dev \
    libpoppler-cpp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configurer python3.11 pour qu'il soit le "python" par défaut
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Mettre à jour pip
RUN python -m pip install --upgrade pip setuptools wheel

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# --- NOUVELLE SOLUTION : INSTALLATION PRÉ-COMPILÉE ---

RUN python -m pip install llama-cpp-python --force-reinstall --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Étape 4 : Copier tout le code source
COPY . .

# Étape 5 : Exposer le port
EXPOSE 7860

# Étape 6 : Commande de lancement
CMD ["python", "scripts/Query_LLM_JINA4.py"]
