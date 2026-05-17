# 🤖 Assistant IA Selmoni — Système RAG Industriel

Système de **Retrieval-Augmented Generation (RAG)** conçu pour interroger des documentations techniques de matériel industriel (SEW, SINAMICS, ROCKWELL...) à l'aide d'un LLM local et d'un moteur de recherche vectorielle basé sur FAISS.

---

## 📋 Table des matières

1. [Architecture du projet](#architecture-du-projet)
2. [Prérequis](#prérequis)
3. [Modèles requis](#modèles-requis)
4. [Build Docker](#build-docker)
5. [Partie 1 — Embedding](#partie-1--embedding)
6. [Partie 2 — Query (Interface de chat)](#partie-2--query-interface-de-chat)
7. [Résumé des commandes](#résumé-des-commandes)

---

## Architecture du projet

### Structure des fichiers source

```
projet/
├── Dockerfile
├── requirements.txt
├── run_app_2.sh          # Lancement de l'embedding
├── run_app_5.sh          # Lancement de l'interface de chat
└── scripts/
    ├── EMBEDDING_MULT.py
    └── Query_LLM_JINA4_bis.py
```

### Structure des dossiers montés (hôte)

Ces dossiers sont montés dans le conteneur via les volumes Docker. Ils doivent exister sur la machine hôte **avant** le lancement des scripts.

```
/home/<user>/
│
├── Documents/
│   ├── models/                         → /app/models  (modèles IA)
│   │   ├── jina-embeddings-v4/         # Modèle d'embedding Jina v4
│   │   ├── qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf  # LLM Qwen2.5 quantisé
│   │   ├── bge-reranker-v2-m3/         # Modèle de reranking cross-encoder
│   │   └── selmoni.png                 # Logo affiché dans l'interface
│   │
│   └── data/                           → /app/data  (données sources)
│       └── parsed/                     # Fichiers JSON issus du parser PDF
│           ├── SEW/
│           │   ├── variateurs/
│           │   │   ├── doc1.json
│           │   │   └── doc2.json
│           │   └── moteurs/
│           │       └── doc3.json
│           ├── SINAMICS/
│           │   └── *.json
│           └── ROCKWELL/
│               └── *.json
│
└── Parcours_Industrie/
    ├── scripts/                        → /app/scripts  (scripts Python)
    │   ├── EMBEDDING_MULT.py
    │   └── Query_LLM_JINA4_bis.py
    │
    └── embeddings/                     → /app/embeddings  (index vectoriels générés)
        ├── embedded_state.json         # Fichier d'état de l'embedding incrémental
        ├── SEW/
        │   ├── SEW.faiss
        │   └── SEW.pkl
        ├── SINAMICS/
        │   ├── SINAMICS.faiss
        │   └── SINAMICS.pkl
        └── ROCKWELL/
            ├── ROCKWELL.faiss
            └── ROCKWELL.pkl
```

> **Note :** Le dossier `embeddings/` est **généré automatiquement** par `EMBEDDING_MULT.py`. Il doit simplement exister vide au premier lancement.

### Format attendu des fichiers JSON (dans `parsed/`)

Chaque fichier `.json` doit respecter le format suivant, issu du parser PDF :

```json
{
  "md5": "abc123...",
  "pages": [
    {
      "page": 1,
      "content": "## Titre de section\n\nContenu de la page en Markdown..."
    },
    {
      "page": 2,
      "content": "..."
    }
  ]
}
```

> L'organisation en sous-dossiers à l'intérieur de `parsed/<MARQUE>/` est libre. Le script détecte automatiquement la marque depuis le **premier niveau** du chemin relatif (ex: `parsed/SEW/...` → marque `SEW`).

---

## Prérequis

- **Docker** avec support **NVIDIA GPU** (`nvidia-container-toolkit` installé)
- Un GPU NVIDIA avec drivers CUDA 12.4 compatibles
- Les modèles IA téléchargés localement (voir section suivante)

Vérifier que Docker voit bien le GPU :

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Modèles requis

Télécharger et placer les modèles dans `/home/<user>/Documents/models/` :

| Modèle | Usage | Source |
|---|---|---|
| `jina-embeddings-v4/` | Embedding des chunks & requêtes | [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) |
| `qwen2.5-7b-instruct-q6_k-*.gguf` | LLM de génération de réponses | [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |
| `bge-reranker-v2-m3/` | Reranking des résultats FAISS | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |


---

## Build Docker

Depuis le dossier racine du projet (là où se trouve le `Dockerfile`) :

```bash
docker build -t selmoni .
```

Cette commande :
- Part de l'image `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- Installe Python 3.11 et toutes les dépendances Python (`requirements.txt`)
- Compile `llama-cpp-python` avec support CUDA 12.4

> Le build peut prendre **10 à 20 minutes** selon la connexion et le CPU de la machine.

---

## Partie 1 — Embedding

### Description

`EMBEDDING_MULT.py` parcourt le dossier `parsed/` à la recherche de fichiers JSON, les découpe en chunks sémantiques (mode HiRAG avec préservation de la hiérarchie des titres), calcule leurs embeddings via le modèle **Jina v4**, puis les indexe dans des bases **FAISS** organisées par marque.

Le script est **incrémental** : un fichier `embedded_state.json` trace les fichiers déjà traités (via leur signature MD5). Seuls les fichiers nouveaux ou modifiés sont ré-embeddés.

### Paramètres clés (dans `EMBEDDING_MULT.py`)

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `MAX_TOKENS` | `512` | Taille maximale d'un chunk en tokens |
| `CHUNK_OVERLAP` | `100` | Overlap entre chunks consécutifs |
| `EMBEDDING_DIM` | `2048` | Dimension des vecteurs Jina v4 |
| `BATCH_SIZE` | `32` | Taille des batchs GPU |

### Lancement

```bash
chmod +x run_app_2.sh
./run_app_2.sh
```

Contenu du script `run_app_2.sh` :

```bash
docker run -it --rm \
   --gpus all \
   -v /home/electhor/Documents/models:/app/models \
   -v /home/electhor/Parcours_Industrie/scripts:/app/scripts \
   -v /home/electhor/Documents/data:/app/data \
   -v /home/electhor/Parcours_Industrie/embeddings:/app/embeddings \
   selmoni \
   python scripts/EMBEDDING_MULT.py
```

### Sortie attendue

```
============================================================
  Embedding incrémental — démarrage
  Source  : /app/data/parsed
  Sortie  : /app/embeddings
============================================================

  JSONs trouvés     : 42
  Déjà embeddés     : 38
  À traiter         : 4

Chargement du tokenizer et du modèle d'embedding...
Modèle prêt.

Chunkisation de tous les nouveaux fichiers...
  [1/4] SEW/variateurs/doc_new.json
    → 87 chunks
  ...

──────────────────────────────────────────────────
  Marque : SEW  (1 nouveau(x) fichier(s))
──────────────────────────────────────────────────
  Calcul des embeddings pour 87 chunks...
  → Sauvegardé : /app/embeddings/SEW/SEW.faiss (1024 vecteurs total)
  ✓ Index SEW : 1024 vecteurs au total

============================================================
  Embedding terminé.
============================================================
```

### Résultat généré

```
embeddings/
├── embedded_state.json     ← mis à jour automatiquement
├── SEW/
│   ├── SEW.faiss           ← index vectoriel FAISS (similarité cosinus IP)
│   └── SEW.pkl             ← métadonnées des chunks (texte, source, page)
├── SINAMICS/
│   ├── SINAMICS.faiss
│   └── SINAMICS.pkl
└── ROCKWELL/
    ├── ROCKWELL.faiss
    └── ROCKWELL.pkl
```

---

## Partie 2 — Query (Interface de chat)

### Description

`Query_LLM_JINA4_bis.py` lance une interface web **Gradio** (port `7860`) permettant d'interroger en langage naturel les documents indexés. Pour chaque question :

1. La requête est encodée avec **Jina v4**
2. Les chunks les plus proches sont récupérés depuis **FAISS**
3. Un **reranker cross-encoder** (BGE v2-m3) affine le classement
4. Le **LLM Qwen2.5-7B** (via `llama-cpp-python`) génère une réponse contextuelle
5. La page source du document PDF original est extraite et affichée dans l'interface

### Paramètres clés (dans `Query_LLM_JINA4_bis.py`)

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.55` | Score minimum FAISS pour retenir un chunk |
| `CHUNKS_FAISS` | `5` | Nombre de chunks bruts envoyés au LLM |
| `CHUNKS_RERANK` | `0` | Nombre de chunks après reranking envoyés au LLM |
| `AVAILABLE_BRANDS` | `["SEW", "SINAMICS", "ROCKWELL"]` | Marques disponibles dans le menu déroulant |

> Pour ajouter une nouvelle marque, il suffit d'ajouter son nom dans `AVAILABLE_BRANDS` et de s'assurer que l'index FAISS correspondant a été généré via l'étape d'embedding.

### Prérequis au lancement

- L'étape d'**embedding** doit avoir été exécutée au moins une fois
- Les fichiers `.faiss` et `.pkl` doivent être présents dans `embeddings/<MARQUE>/`
- Les fichiers PDF sources doivent être accessibles dans `/app/data/` pour l'affichage de la page source

### Lancement

```bash
chmod +x run_app_5.sh
./run_app_5.sh
```

Contenu du script `run_app_5.sh` :

```bash
docker run -it --rm \
  --gpus all \
  -v /home/electhor/Documents/models:/app/models \
  -v /home/electhor/Parcours_Industrie/scripts:/app/scripts \
  -v /home/electhor/Parcours_Industrie/embeddings:/app/embeddings \
  -v /home/electhor/Documents/data:/app/data \
  -p 7860:7860 \
  selmoni \
  python scripts/Query_LLM_JINA4_bis.py
```

### Accès à l'interface

Une fois le conteneur démarré, ouvrir dans un navigateur :

```
http://localhost:7860
```


---

## Résumé des commandes

```bash
# 1. Build de l'image Docker (une seule fois, ou après modification du Dockerfile)
docker build -t selmoni .

# 2. Générer / mettre à jour les index d'embedding
./run_app_2.sh

# 3. Lancer l'interface de chat
./run_app_5.sh
```

---

## 🛠️ Dépannage

**Le GPU n'est pas détecté dans le conteneur**
Vérifier que `nvidia-container-toolkit` est installé et que le daemon Docker est redémarré après son installation.

```bash
sudo systemctl restart docker
```

**`CUDA out of memory` pendant l'embedding**
Réduire `BATCH_SIZE` dans `EMBEDDING_MULT.py` (essayer `8` ou `4`).

**Aucun chunk retourné lors d'une requête**
Vérifier que les index `.faiss` et `.pkl` existent bien dans le dossier `embeddings/<MARQUE>/`. Relancer l'embedding si nécessaire.

**L'interface Gradio n'est pas accessible**
S'assurer que le port `7860` n'est pas utilisé par un autre processus sur l'hôte, et que le flag `-p 7860:7860` est bien présent dans `run_app_5.sh`.

---

