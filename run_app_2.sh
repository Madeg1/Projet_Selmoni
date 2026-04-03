#!/bin/bash

docker run -it --rm  \
   --gpus all \
   -v /home/electhor/Documents/models:/app/models \
   -v /home/electhor/Parcours_Industrie/scripts:/app/scripts \
   -v /home/electhor/Documents/data:/app/data \
   -v /home/electhor/Parcours_Industrie/embeddings:/app/embeddings \
   selmoni \
   python scripts/EMBEDDING_MULT.py

