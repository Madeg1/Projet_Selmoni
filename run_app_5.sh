#!/bin/bash

# Lancement du conteneur
docker run -it --rm \
  --gpus all \
  -v /home/electhor/Documents/models:/app/models \
  -v /home/electhor/Parcours_Industrie/scripts:/app/scripts \
  -v /home/electhor/Parcours_Industrie/embeddings:/app/embeddings \
  -v /home/electhor/Documents/data:/app/data \
  -p 7860:7860 \
  selmoni \
  python scripts/Query_LLM_JINA4_bis.py


