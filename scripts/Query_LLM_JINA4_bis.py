import faiss
import os
import re
import pickle
import numpy as np
import time
import sys
import threading
import base64
import torch
from transformers import AutoModel
from llama_cpp import Llama
import gradio as gr
from gradio_pdf import PDF
import fitz
from transformers import AutoModelForSequenceClassification
import urllib.parse

#---Fonction pour encoder l'image---
def encode_image(image_path):
    """Convertit une image en chaîne base64 pour l'intégration HTML directe"""
    try:
        if not os.path.exists(image_path):
            return ""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Erreur lors du chargement de l'image : {e}")
        return ""


# --- Fonction pour extraire la page cible + contexte ---
def extract_pages_with_context(pdf_path, target_page_num, output_path, context=1):
    """
    Extrait la page cible + 'context' pages avant et après.
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Conversion en index (page 1 -> index 0)
        target_idx = int(target_page_num) - 1
        
        if target_idx < 0 or target_idx >= total_pages:
            print(f"Page {target_page_num} hors limites.")
            return None
            
        # On s'assure de ne pas descendre sous 0 ou dépasser le max
        start_idx = max(0, target_idx - context)
        end_idx = min(total_pages - 1, target_idx + context)
        
        print(f"Extraction des pages {start_idx+1} à {end_idx+1} (Cible: {target_page_num})")

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_idx, to_page=end_idx)
        
        new_doc.save(output_path)
        new_doc.close()
        doc.close()
        
        return output_path
    except Exception as e:
        print(f"Erreur CRITIQUE extraction PDF: {e}")
        return None
        
        
#-------------------------------------
#           CONFIGURATION
#-------------------------------------

SIMILARITY_THRESHOLD = 0.55
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


EMBEDDING_MODEL_NAME = '/app/models/jina-embeddings-v4'
RERANKER_MODEL_NAME = '/app/models/jina-reranker-v2-base-multilingual'
LLM_MODEL_PATH = '/app/models/qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf'

BASE_EMBEDDINGS_PATH = '/app/embeddings'
AVAILABLE_BRANDS = ["SEW", "SINAMICS", "ROCKWELL"]

LOADED_RESOURCES = {}

PATH_LOGO = '/app/models/selmoni.png'
logo_base64 = encode_image(PATH_LOGO)
img_src = f"data:image/png;base64,{logo_base64}" if logo_base64 else "https://via.placeholder.com/60"

#-------------------------------------
#    CLASSE WRAPPER RERANKER
#-------------------------------------
class JinaReranker:
    def __init__(self, model_path, device):
        print(f" Chargement du Reranker Jina sur : {device.upper()}")
        try:
            # On charge le modèle en fp16 pour économiser la VRAM
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            ).to(device)
            self.model.eval()
        except Exception as e:
            print(f" Erreur chargement Reranker: {e}")
            sys.exit(1)

    def compute_scores(self, query, texts):
        """Calcule le score de pertinence entre la question et chaque chunk"""
        with torch.no_grad():
            # Le modèle attend une liste de paires [question, document]
            pairs = [[query, text] for text in texts]
            # La méthode compute_score est intégrée dans le code distant de Jina
            scores = self.model.compute_score(pairs)
            
            # Conversion propre en liste de floats
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy().tolist()
            elif isinstance(scores, np.ndarray):
                scores = scores.tolist()
            elif isinstance(scores, float):
                scores = [scores]
                
            return scores

#-------------------------------------
#    CLASSE WRAPPER JINA 
#-------------------------------------
class JinaEmbedder:#utilisation d'une classe pour le modèle Jina
    
    def __init__(self, model_path, device):
        print(f" Chargement du modèle Jina sur : {device.upper()}")
        try:
          
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
            self.model.eval()
        except Exception as e:
            print(f" Erreur chargement Jina: {e}")
            sys.exit(1)
    #Embedding
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str): texts = [texts]#s'assure que l'entrée est une liste
        
        with torch.no_grad():
           
            batch_output = self.model.encode_text(texts, task = "retrieval") #appel à la méthode d'encodage Jina
            
            if isinstance(batch_output, list):# gère le cas où la sortie est une liste de tenseurs
                embeddings = np.array([t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in batch_output])#conversion en numpy array
            elif isinstance(batch_output, torch.Tensor):# gère le cas où la sortie est un tenseur unique
                embeddings = batch_output.detach().cpu().numpy()#conversion en numpy array
            else:
                embeddings = np.array(batch_output)# conversion directe en numpy array
        if normalize_embeddings:
            faiss.normalize_L2(embeddings)# normalisation L2 des embeddings
            
        return embeddings.astype('float32')# conversion finale en float32

#---------------------------------------
#      CHARGEMENT DES RESSOURCES
#---------------------------------------


model = JinaEmbedder(EMBEDDING_MODEL_NAME, DEVICE)
reranker = JinaReranker(RERANKER_MODEL_NAME, DEVICE)

def get_brand_resources(brand_name):
    """
    Charge l'index FAISS et les chunks pour une marque spécifique si ce n'est pas déjà fait.
    Retourne (index, text_chunks) ou lève une exception.
    """
    if brand_name in LOADED_RESOURCES:
        return LOADED_RESOURCES[brand_name]['index'], LOADED_RESOURCES[brand_name]['chunks']
    
    print(f" Chargement des ressources pour : {brand_name}...")
    
    brand_path = os.path.join(BASE_EMBEDDINGS_PATH, brand_name)
    faiss_path = os.path.join(brand_path, f"{brand_name}.faiss")
    pkl_path = os.path.join(brand_path, f"{brand_name}.pkl")     

    try:
        index = faiss.read_index(faiss_path)
        with open(pkl_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # Sauvegarde en cache mémoire
        LOADED_RESOURCES[brand_name] = {'index': index, 'chunks': chunks}
        print(f" Ressources {brand_name} chargées.")
        return index, chunks
    except Exception as e:
        print(f" Erreur chargement {brand_name}: {e}")
        raise e

print("Ressources prêtes.")


#----- Chargement du LLM -----
print(f" Chargement du LLM depuis : {LLM_MODEL_PATH}")
llm = Llama(
    model_path=LLM_MODEL_PATH, 
    n_ctx=8192,# Taille du contexte
    n_gpu_layers=-1,      # Utilisation de toutes les couches GPU disponibles
    n_batch=512,          # Taille du batch
    f16_kv=True,          # Utilisation de la précision float16 pour les clés/valeurs
    n_threads=os.cpu_count(),# Nombre de threads CPU
    flash_attn=True, # Activation de l'attention flash
    use_mmap=False, # Désactivation de la mémoire mappée
)
print("Système prêt.")

#------------------------------
#   LOGIQUE DE RECHERCHE 
#------------------------------

def search(query, index_obj, chunks_list, k=5):
    print(f"Recherche pour: '{query}'")
    start_search_time = time.perf_counter()
    
    query_embedding = model.encode([query], normalize_embeddings=True)# embedding de la requête
    
    similarities, indices = index_obj.search(query_embedding, k) # recherche dans l'index FAISS
    
    search_duration = time.perf_counter() - start_search_time
    
    results = []
    for i in range(k):
        chunk_index = indices[0][i]# Indice du chunk
        sim = similarities[0][i]# Similarité correspondante
        
        if chunk_index < len(chunks_list):# Vérifie que l'indice est valide
            chunk_data = chunks_list[chunk_index]
            results.append(chunk_data)
        
    return results, similarities[0], search_duration

#-------------------------------------------------
#   FONCTIONS GRADIO & GENERATION
#-------------------------------------------------
def clear_chat():
    """Réinitialise l'historique, le champ de texte et la vue PDF"""
    default_pdf_html = '<div style="height: 100%; display: flex; align-items: center; justify-content: center; color: #94a3b8;">Le document source s\'affichera ici après la recherche.</div>'
    
    return (
        [],               # Vide le composant Chatbot 
        "",               # Vide la barre de question
        default_pdf_html  # Remet le message par défaut à la place du PDF
    )


def stop_server():
    print("Arrêt du serveur...")
    threading.Thread(target=lambda: (sys.exit())).start() # Lance l'arrêt du serveur dans un thread séparé
    return "Serveur arrêté."

#-------------------------------------------
#            Détection de mot clés dans les meilleurs chunks
#-----------------------------------------------

def find_best_matching_chunk(llm_answer, chunks):
    if not chunks or not llm_answer:
        return chunks[0] if chunks else None
    
    # Pas de calcul si le LLM n'a rien trouvé
    if "introuvable" in llm_answer.lower():
        return chunks[0]
    
    # Embedding de la réponse
    answer_embedding = model.encode([llm_answer], normalize_embeddings=True)
    
    # Embedding des chunks candidats
    chunk_texts = [c.get('llm_context', c['text']) for c in chunks]
    chunks_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
    
    # Similarité cosinus (les embeddings sont déjà normalisés → produit scalaire suffit)
    similarities = np.dot(chunks_embeddings, answer_embedding.T).flatten()
    best_idx = int(np.argmax(similarities))
    
    return chunks[best_idx]
    
    

SEW_NOMENCLATURE_HINT = """
RÈGLES DE DÉCODAGE SEW (APPLICATION OBLIGATOIRE) :
1. Pour trouver un accessoire, convertis la référence en entier pur (ex: MCC91A-0025 -> La puissance cible est 25).
2. Convertis les plages des tableaux en entiers purs (ex: "0010 - 0070" devient "de 10 à 70").
3. Compare la puissance cible à chaque plage.
"""
def needs_nomenclature_hint(brand: str, query: str) -> bool:
    """Détecte si la query implique un décodage de référence SEW."""
    if brand != "SEW":
        return False
    # Détecte un pattern de référence type MCC91A-XXXX-XEX
    return bool(re.search(r"MCC\w+[-–]\d{4}[-–]\d+E\d+", query, re.IGNORECASE))
    
def rewrite_query(history, current_query, llm):
    # On inclut l'historique dans le prompt uniquement s'il existe
    hist_text = f"Historique: {history[-1]}\n" if history else ""
    
    prompt = (
        "<|im_start|>system\nTu es un reformulateur de requête pour un moteur de recherche technique. "
        "Ta tâche est de nettoyer le dernier message de l'utilisateur pour optimiser la recherche.\n"
        "1. Si tu vois une référence produit très longue (ex: MCC91A-0025-5E3-4-000/CSO/CFX11A-N), supprime les suffixes inutiles de communication (/CSO, /CFX...) et ne garde que la famille et la puissance.\n"
        "2. Si l'utilisateur dit juste 'oui' ou un mot court, utilise l'historique pour formuler une question complète.\n"
        "3. Ne réponds SURTOUT PAS à la question technique. Renvoie UNIQUEMENT la phrase de recherche nettoyée.<|im_end|>\n"
        f"{hist_text}"
        f"Dernier message: {current_query}\n"
        "<|im_start|>assistant\nRequête optimisée :"
    )
    
    response = llm(prompt, max_tokens=50, temperature=0.1, stop=["\n", "<|im_end|>"])
    return response['choices'][0]['text'].strip()    
    
    
def generate_response(brand,query,history=None, max_context_tokens=8000):
    if history is None:
        history = []
    
    if not brand:
        return "Veuillez sélectionner une marque.", ""    
    
    print(f"Début génération | Marque: {brand} | Query: '{query}'")
    
    # ==========Contextualisation de la recherche FAISS===============
    # On reformule TOUJOURS la question, qu'il y ait un historique ou non
    print(" Nettoyage de la requête par le LLM...")
    search_query = rewrite_query(history, query, llm)
    print(f"Question reformulée pour FAISS : '{search_query}'")
    # =========================================================
    
    
    # Chargement dynamique des ressources de la marque
    try:
        current_index, current_chunks = get_brand_resources(brand)
    except Exception as e:
        return f"Erreur critique lors du chargement de la marque {brand} : {str(e)}", ""
    
    # Recherche large (FAISS)
    faiss_chunks, similarities, search_time = search(search_query, current_index, current_chunks, k=30)
    if not faiss_chunks:
        return "Aucune information trouvée dans la base.", ""
    
    #RERANKING
    print(f"\n Reranking de {len(faiss_chunks)} chunks...")
    start_rerank = time.perf_counter()
    
    texts_to_rerank = [c.get('llm_context', c['text']) for c in faiss_chunks]
    
    try:
        rerank_scores = reranker.compute_scores(search_query, texts_to_rerank)
    except Exception as e:
        print(f"Erreur Reranker ({e}). Fallback sur FAISS.")
        rerank_scores = similarities
        
    rerank_time = time.perf_counter() - start_rerank
    print(f" Reranking terminé en {rerank_time:.2f}s")

    # On associe chaque chunk à son score de reranking et on trie du meilleur au pire
    scored_chunks = list(zip(faiss_chunks, rerank_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # On isole les chunks triés pour la suite du pipeline
    relevant_chunks = [chunk for chunk, score in scored_chunks]
    
    #--- GESTION DU PDF ---
    best_chunk = scored_chunks[0][0]
    best_score = scored_chunks[0][1]
    filename = best_chunk.get('source', '')
    page_num = best_chunk.get('page', 1)
    
    base_folder = "/app/data"
    full_path = os.path.join(base_folder, filename)
    
    # Fichier temporaire
    temp_pdf_path = f"/tmp/context_{page_num}_{int(time.time())}.pdf"
    pdf_html_output = '<div style="text-align:center; color:#94a3b8;">Aucun document à afficher.</div>'
    
    if os.path.exists(full_path) and best_score>0:
        # On extrait 3 pages : [Page Avant] - [Page Cible] - [Page Après]
        extracted_path = extract_pages_with_context(full_path, page_num, temp_pdf_path, context=1)
        
        if extracted_path:
            try:
                # On convertit le fichier PDF (les 3 pages) en texte Base64
                with open(extracted_path, "rb") as f:
                    pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                pdf_data_url = f"data:application/pdf;base64,{pdf_base64}#page=2"
                
                pdf_html_output = f"""
                    <iframe src="{pdf_data_url}" width="100%" height="800px" style="border:none; border-radius:8px;">
                    </iframe>
                """
            except Exception as e:
                print(f"Erreur Base64: {e}")
                pdf_html_output = f"Erreur : {e}"
    
    
    context_texts = []
    sources_formatted = [] 
    injected_chunks_list = []
    current_token_count = 0
    seen_hashes = set()
    
    # --- NOUVEAU : On définit notre objectif ---
    MAX_CHUNKS_TO_INJECT = 5
    chunks_injected = 0

    system_overhead = len(llm.tokenize(b"System prompt template overhead...")) + 500 
    limit = 8192 - system_overhead - 512 

    print(f"\n{'='*60}")
    print(f"ANALYSE DES CHUNKS POUR : {query}")
    print(f"{'='*60}")

    for i, (chunk_data, sim) in enumerate(scored_chunks):
        
        # --- NOUVEAU : On s'arrête si on a nos 5 bons chunks ---
        if chunks_injected >= MAX_CHUNKS_TO_INJECT:
            print(f"\n [Stop] Objectif atteint : {MAX_CHUNKS_TO_INJECT} chunks uniques injectés.")
            break
            
        page_num = chunk_data.get('page', '?') 
        filename = chunk_data['source']
        text_content = chunk_data.get('llm_context', chunk_data['text'])
        
        source_identifier = f"{filename} (Page {page_num})"

        # Filtre Anti-doublon
        content_hash = hash(text_content)
        if content_hash in seen_hashes:
            print(f" [Doublon ignoré] {source_identifier} (déjà injecté)")
            continue

        # Filtre Pertinence
        if sim < 0: 
            print(f" [Rejeté - Score négatif] {source_identifier} (Score: {sim:.4f})")
            continue
        
        chunk_tokens = llm.tokenize(text_content.encode("utf-8"))
        num_tokens = len(chunk_tokens)

        # Filtre Contexte (Tokens)
        if current_token_count + num_tokens > max_context_tokens:
            print(f" [Stop - Contexte plein] {source_identifier} ne rentre pas.")
            break 
            
        # ========================================================
        # ZONE DE VALIDATION 
        # ========================================================
        seen_hashes.add(content_hash)
        context_texts.append(text_content)
        injected_chunks_list.append(chunk_data) #on garde le chunk entier
        chunks_injected += 1 # On incrémente notre compteur
        
        # Formatage 
        source_line = f"{filename} — Page {page_num} (Score: {sim:.4f})"
        if len(sources_formatted) < 3 :
            sources_formatted.append(source_line)
        
        current_token_count += num_tokens

        print(f"\n [Validé {chunks_injected}/{MAX_CHUNKS_TO_INJECT}] Chunk {i+1} original | Sim: {sim:.4f} | {source_identifier}")
        print("-" * 60)
        print(text_content) 
        print("-" * 60)

    if not context_texts:
        return "Documents trouvés mais pertinence trop faible (aucun chunk n'a été retenu).",""
    

    
    # Construction du contexte final
    context = "\n\n---\n\n".join(context_texts)
    
    # Création de la chaîne avec sauts de ligne pour les sources
    source_list_str = "\n".join(sources_formatted)
    
    print(f"\n Résumé Contexte : {len(context_texts)} chunks injectés.")

    hint = SEW_NOMENCLATURE_HINT if needs_nomenclature_hint(brand, query) else ""
    
    prompt_template = (
        "<|im_start|>system\n"
        "Tu es un assistant expert technique chez Selmoni. Tu es strictement factuel.\n"
        + hint +
        "Utilise UNIQUEMENT le contexte ci-dessous ET l'historique pour répondre.\n"
        "ATTENTION : Le contexte contient des tableaux. Lis attentivement les en-têtes.\n"
        "RÈGLE 1 : Tu DOIS TOUJOURS réfléchir étape par étape. Écris TOUTE ton analyse du texte et tes calculs entre des balises <reflexion> et </reflexion> au tout début de ta réponse.\n"
        "RÈGLE 2 : Dans ta <reflexion>, si tu dois choisir une valeur dans une plage de type SEW (ex: 0010 - 0070), retire les zéros initiaux et écris explicitement l'inégalité mathématique (ex: 10 <= 25 <= 70) pour prouver ton choix.\n"
        "RÈGLE 3 : N'invente JAMAIS d'informations. Si une donnée est absente du contexte, déclare explicitement qu'elle est introuvable.\n"
        "RÈGLE 4 : En dehors des balises de réflexion, formule ta réponse finale. Cette réponse DOIT être une phrase complète, détaillée, reprenant les mots-clés exacts (ex: 'La puissance absorbée de la borne STO est de 150 mW.').\n"
        "RÈGLE 5 (INTERACTION) : Si l'information est vraiment introuvable, demande-toi si l'utilisateur n'a pas utilisé un synonyme (ex: 'consommation' au lieu de 'puissance'). Si c'est le cas, pose-lui une question courte et polie pour clarifier (Ex: 'Information introuvable. Cherchiez-vous la puissance absorbée ?').\n"
        "<|im_end|>\n"
    )
    
    for old_query, old_response in history[-3:]:
        # On supprime la partie "Sources utilisées" de l'ancienne réponse pour économiser des tokens
        clean_old_response = old_response.split("\n\n---")[0].strip()
        prompt_template += f"<|im_start|>user\n{old_query}<|im_end|>\n"
        prompt_template += f"<|im_start|>assistant\n{clean_old_response}<|im_end|>\n"

    # On ajoute le contexte actuel et la nouvelle question
    prompt_template += f"<|im_start|>user\nCONTEXTE:\n{context}\n\nQUESTION:\n{query}<|im_end|>\n<|im_start|>assistant\n"


    print(" Génération de la réponse par le LLM...")
    start_llm = time.perf_counter()
    
    response = llm(
        prompt=prompt_template,
        max_tokens=512, # 512 est amplement suffisant maintenant
        stop=["<|im_end|>", "<|im_start|>", "user\n", "CONTEXTE:\n"], 
        temperature=0.1,
        repeat_penalty=1.15,      
        frequency_penalty=0.2,    
        presence_penalty=0.2      
    )
    
    duration = time.perf_counter() - start_llm
    raw_answer = response['choices'][0]['text'].strip() 
    
    print(f"\n--- RÉPONSE BRUTE DU LLM ---\n{raw_answer}\n----------------------------\n")
    print(f" Réponse générée en {duration:.2f}s")
    
    # 1. On cherche la page PDF en utilisant TOUTE la réponse brute (réflexion incluse) pour maximiser le "match" sémantique
    target_chunk = find_best_matching_chunk(raw_answer, injected_chunks_list)
    
    # 2. On nettoie la réponse pour l'utilisateur (on supprime ce qui est entre <reflexion> et </reflexion>)
    clean_answer = re.sub(r'<reflexion>.*?</reflexion>', '', raw_answer, flags=re.DOTALL).strip()
    
    # Sécurité : si le modèle n'a pas mis de balises, on affiche tout
    display_answer = clean_answer if clean_answer else raw_answer
    
    if target_chunk:
        filename = target_chunk.get('source', '')
        page_num = target_chunk.get('page', 1)
        print(f"Page déduite par chevauchement lexical : {page_num} (Fichier: {filename})")
        
        base_folder = "/app/data"
        full_path = os.path.join(base_folder, filename)
        temp_pdf_path = f"/tmp/context_{page_num}_{int(time.time())}.pdf"
        
        # On s'assure que le fichier existe avant d'extraire
        if os.path.exists(full_path):
            print(f"Extraction avec contexte (3 pages) pour la page {page_num}...")
            extracted_path = extract_pages_with_context(full_path, page_num, temp_pdf_path, context=1)
            
            if extracted_path:
                try:
                    with open(extracted_path, "rb") as f:
                        pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    pdf_data_url = f"data:application/pdf;base64,{pdf_base64}#page=2"
                    pdf_html_output = f"""
                        <iframe src="{pdf_data_url}" width="100%" height="800px" style="border:none; border-radius:8px;">
                        </iframe>
                    """
                except Exception as e:
                    print(f"Erreur Base64: {e}")
                    pdf_html_output = f"Erreur d'affichage PDF : {e}"
        else:
             print(f"Fichier source introuvable : {full_path}")
    
    final_output = (
        f"{display_answer}\n\n"
        f"---\n"
        f"**Sources utilisées :**\n"
        f"{source_list_str}\n\n"
        f"*(Recherche: {search_time:.2f}s | Reranking: {rerank_time:.2f}s | Génération: {duration:.2f}s)*"
    )
    
    return final_output, pdf_html_output


def chat_interaction(brand, query, history):
    """
    Gère l'interaction avec le composant Chatbot.
    history est une liste de tuples : [(user_msg, bot_msg), ...]
    """
    history = history or [] # Initialise l'historique si vide
    
    if not brand:
        history.append((query, " Veuillez sélectionner une marque."))
        return history, "", '<div style="text-align:center; color:#94a3b8;">Aucun document à afficher.</div>'
    
    if not query.strip():
        return history, "", '<div style="text-align:center; color:#94a3b8;">Veuillez poser une question.</div>'

    # Appel de votre fonction de génération actuelle
    answer_text, pdf_html = generate_response(brand, query,history)
    
    # Ajout de l'échange actuel à l'historique visuel
    history.append((query, answer_text))
    
    # On retourne : l'historique mis à jour, une chaîne vide (pour effacer la barre de saisie), et le visualiseur PDF
    return history, "", pdf_html



#------Interface Gradio-------
css_style = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    body { 
        background-color: #f8fafc; 
        font-family: 'Inter', sans-serif; 
        color: #334155;
    }
    .gradio-container { 
        max-width: 1400px !important; 
        margin: 0 auto; 
        padding-top: 30px; 
    }
    
    /* En-tête */
    .header-container { 
        display: flex; 
        align-items: center; 
        gap: 20px; 
        margin-bottom: 2em; 
        padding-bottom: 20px;
        border-bottom: 2px solid #e2e8f0;
    }
    .logo_img { height: 50px; width: auto; }
    .title-text { 
        font-size: 2.2em; 
        font-weight: 700; 
        color: #0f172a; 
        margin: 0; 
    }

    /* Colonnes principales */
    .main-row { gap: 30px; }
    .chat-col, .pdf-col {
        background-color: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
        height: fit-content;
    }
    
    /* Chatbot */
    #chatbot-component {
        height: 600px !important;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        background-color: #f8fafc;
    }
    .message.user { 
        background-color: #3b82f6 !important; 
        color: white !important; 
        border-bottom-right-radius: 0 !important;
    }
    .message.bot { 
        background-color: white !important; 
        border: 1px solid #e2e8f0 !important;
        border-bottom-left-radius: 0 !important;
        color: #334155 !important;
    }

    /* Zone de saisie */
    .input-row {
        margin-top: 20px;
        align-items: stretch;
        gap: 10px;
    }
    .question-box textarea {
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 1em;
        resize: none;
        transition: border-color 0.2s;
        background-color: white;
    }
    .question-box textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    .blue-btn { 
        background-color: #3b82f6 !important; 
        color: white !important; 
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600;
        font-size: 1em;
        transition: background-color 0.2s;
    }
    .blue-btn:hover { background-color: #2563eb !important; }

    /* Boutons du bas */
    .actions-row { margin-top: 20px; justify-content: space-between; }
    .secondary-btn {
        background-color: white !important;
        color: #64748b !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        font-weight: 600;
        transition: all 0.2s;
    }
    .secondary-btn:hover {
        background-color: #f1f5f9 !important;
        color: #334155 !important;
    }
    .stop-btn {
        background-color: #fee2e2 !important;
        color: #dc2626 !important;
        border: 1px solid #fecaca !important;
    }
    .stop-btn:hover {
        background-color: #fecaca !important;
        color: #b91c1c !important;
    }

    /* Titres de section */
    .section-title {
        font-size: 1.2em;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 15px;
        display: block;
    }

    /* Pied de page */
    .footer {
        text-align: center; 
        color: #94a3b8; 
        font-size: 0.85em; 
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }
"""

with gr.Blocks(title="Assistant IA Selmoni", css=css_style) as interface:
    gr.HTML(f"""
    <div style="display: flex; align-items: center; gap: 20px; padding: 10px; margin-bottom: 10px; border-bottom: 2px solid #0056b3;">
        <img src="{img_src}" style="height: 50px;">
        <h1 style="margin: 0; font-family: sans-serif; color: #1e293b;">Assistant IA Selmoni</h1>
    </div>
    """)
    
    with gr.Row(elem_classes="main-row"):
        # --- COLONNE GAUCHE : CHATBOT & INPUT ---
        with gr.Column(scale=1, elem_classes="chat-col"):
            
            brand_selector = gr.Dropdown(
                choices=AVAILABLE_BRANDS,
                value=AVAILABLE_BRANDS[0], 
                label="Sélectionnez la marque",
                interactive=True
            )
            
            
            chatbot = gr.Chatbot(
                label="Conversation avec l'IA", 
                height=500,
                show_copy_button=True
            )
            
            # Zone de saisie et bouton sur la même ligne
            with gr.Row(elem_classes="input-row"):
                question = gr.Textbox(
                    show_label=False, 
                    placeholder="Ex: Que faire lorsque le défaut 11.9 intervient ?",
                    lines=3,           # Définit la hauteur de base à 3 lignes
                    max_lines=10,      # Permet à la boîte de s'agrandir jusqu'à 10 lignes si le texte est long
                    scale=4, 
                    elem_classes="question-box", # Applique votre joli CSS personnalisé
                    container=False    # Enlève le cadre superflu de Gradio
                )
                ask_button = gr.Button("Envoyer", variant="primary", elem_classes="blue-btn", scale=1)
            
            with gr.Row():
                clear_button = gr.Button("Nouvelle conversation", variant="secondary")
                stop_button = gr.Button("Arrêter le serveur", variant="secondary", elem_id="stop-btn")
            
            gr.Markdown("---")
            gr.HTML("<div style='text-align:center; color:#94a3b8; font-size: 0.8em;'>© 2026 Selmoni - Système RAG Interne</div>")
        
        # --- COLONNE DROITE : VISUALISATION PDF (Inchangée) ---
        with gr.Column(scale=1, elem_classes="pdf-col"):
            gr.Markdown("### Document Source (Page extraite)")
            pdf_viewer = gr.HTML(
                value='<div style="height: 100%; display: flex; align-items: center; justify-content: center; color: #94a3b8;">Le document source s\'affichera ici après la recherche.</div>'
            )
            
    # --- GESTION DES ÉVÉNEMENTS ---
    # Quand on appuie sur "Entrée"
    question.submit(
        fn=chat_interaction, 
        inputs=[brand_selector, question, chatbot],
        outputs=[chatbot, question, pdf_viewer] # Met à jour le chat, vide l'input, met à jour le PDF
    )
    brand_selector.change(
        fn=clear_chat, 
        inputs=None, 
        outputs=[chatbot, question, pdf_viewer]
    )
    # Quand on clique sur le bouton "Envoyer"
    ask_button.click(
        fn=chat_interaction, 
        inputs=[brand_selector, question, chatbot],
        outputs=[chatbot, question, pdf_viewer]
    )
    
    stop_button.click(fn=stop_server, inputs=None, outputs=None)
    
    clear_button.click(
        fn=clear_chat, 
        inputs=None, 
        outputs=[chatbot, question, pdf_viewer]
    )




if __name__ == "__main__":
    interface.launch(
        share=False, 
        server_name="0.0.0.0", 
        allowed_paths=["/app/models", "/app/data", "/app/embeddings","/tmp"]
    )
