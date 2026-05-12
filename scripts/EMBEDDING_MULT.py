import json
import numpy as np
import faiss
import pickle
import os
import torch
import ftfy
import gc
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import re

# ======================
# CONFIG
# ======================

PARSED_ROOT      = "/app/data/parsed"       # Miroir du parser : parsed/SEW/..., parsed/SINAMICS/..., etc.
EMBEDDINGS_ROOT  = "/app/embeddings"        # Sortie : embeddings/SEW/SEW.faiss + SEW.pkl, etc.
STATE_FILE       = "/app/embeddings/embedded_state.json"  # Suivi des fichiers déjà embeddés

EMBEDDING_MODEL_NAME = "/app/models/jina-embeddings-v4"
MAX_TOKENS     = 512
CHUNK_OVERLAP  = 100
EMBEDDING_DIM  = 2048
BATCH_SIZE     = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# STATE 
# ======================

def load_state() -> dict:
    """
    Retourne le dict des fichiers JSON déjà embeddés.
    Format : { "chemin/relatif/doc.json": "md5_du_json" }
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def json_signature(json_path: Path) -> str:
    """
    Utilise le md5 stocké dans le JSON (mis là par le parser)
    comme signature, pour éviter de re-lire le PDF.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("md5", str(os.path.getmtime(json_path)))
    except Exception:
        return str(os.path.getmtime(json_path))


# ======================
# DÉCOUVERTE
# ======================

def get_brand_from_path(json_path: Path) -> str:
    """
    Extrait le nom de marque depuis le chemin.
    Ex: /app/data/parsed/SEW/variateurs/doc.json -> "SEW"
    """
    parsed_root = Path(PARSED_ROOT).resolve()
    relative    = json_path.resolve().relative_to(parsed_root)
    return relative.parts[0].upper()


def collect_json_files() -> dict[str, list[Path]]:
    """
    Parcourt PARSED_ROOT récursivement et regroupe les JSON par marque.
    Retourne: { "SEW": [Path, ...], "SINAMICS": [...], ... }
    """
    brand_files: dict[str, list[Path]] = {}
    for json_path in sorted(Path(PARSED_ROOT).rglob("*.json")):
        brand = get_brand_from_path(json_path)
        brand_files.setdefault(brand, []).append(json_path)
    return brand_files


# ======================
# CHUNKING & TABLEAUX (MODE HiRAG)
# ======================

def split_into_markdown(text: str, tokenizer, max_tokens: int, overlap: int = 100) -> list[str]:
    """Découpe un bloc de texte s'il dépasse la limite de tokens."""
    lines  = text.split("\n")
    chunks = []
    current_chunk_tokens = []

    for line in lines:
        line_tokens = tokenizer.encode(line + "\n", add_special_tokens=False)

        if len(line_tokens) > max_tokens:
            if current_chunk_tokens:
                chunks.append(tokenizer.decode(current_chunk_tokens, skip_special_tokens=True))
                current_chunk_tokens = []
            for i in range(0, len(line_tokens), max_tokens):
                chunks.append(tokenizer.decode(line_tokens[i: i + max_tokens], skip_special_tokens=True))
            continue

        if len(current_chunk_tokens) + len(line_tokens) <= max_tokens:
            current_chunk_tokens.extend(line_tokens)
        else:
            chunks.append(tokenizer.decode(current_chunk_tokens, skip_special_tokens=True))
            current_chunk_tokens = current_chunk_tokens[-overlap:] if overlap > 0 else []
            current_chunk_tokens.extend(line_tokens)

    if current_chunk_tokens:
        last_chunk = tokenizer.decode(current_chunk_tokens, skip_special_tokens=True)
        if last_chunk.strip():
            chunks.append(last_chunk)

    return chunks


def process_page_hirag(content: str, source: str, page_num: int, tokenizer) -> list[dict]:
    """
    Parcourt la page ligne par ligne en mémorisant la hiérarchie des titres (H1, H2, H3...).
    Sépare proprement les blocs de texte et les tableaux sans doublons.
    Fusionne le texte introductif avec le tableau qui le suit.
    """
    lines = content.split('\n')
    chunks = []
    
    active_headers = {}  # Stocke le niveau du titre -> Texte du titre (ex: {2: "## 4.23 Accessoires"})
    current_text_block = []
    current_table = []
    in_table = False
    table_intro = ""  # <--- NOUVEAU : Mémoire tampon pour l'intro du tableau
    
    def get_hierarchy_context() -> str:
        """Génère le chemin sémantique parent pour le chunk."""
        if not active_headers:
            return ""
        sorted_levels = sorted(active_headers.keys())
        hierarchy = " > ".join([active_headers[lvl].replace('#', '').strip() for lvl in sorted_levels])
        return f"[CONTEXTE : {hierarchy}]\n\n"

    def flush_text():
        nonlocal current_text_block
        if not current_text_block:
            return
    
        raw_text = "\n".join(current_text_block).strip()
    
    
        lines_with_content = [
            l for l in raw_text.split('\n')
            if l.strip() and not l.strip().startswith('#')
        ]
        if not lines_with_content:
            current_text_block = []
            return
    
        if raw_text:
            hierarchy_prefix = get_hierarchy_context()
            sub_chunks = split_into_markdown(raw_text, tokenizer, MAX_TOKENS, CHUNK_OVERLAP)
            for sc in sub_chunks:
                final_text = hierarchy_prefix + sc
                chunks.append({
                    "text": final_text,
                    "llm_context": final_text,
                    "source": source,
                    "page": page_num,
                    "is_table": False,
                    "synthetic": False
                })
        current_text_block = []

    def flush_table():
        def parse_table_to_prose(header_line: str, separator_line: str, data_rows: list[str]) -> str:
            try:
                headers = [h.strip() for h in header_line.strip().strip('|').split('|')]
                sentences = []

                for row in data_rows:
                    cells = [c.strip() for c in row.strip().strip('|').split('|')]
                    if len(cells) < 2 or not cells[0]:
                        continue

                    label = cells[0]

                    # Détecte si la colonne 1 ressemble à une unité (courte, pas de chiffre isolé)
                    # Ex: "W", "A", "Hz", "kW", "V", "kg", "s", "min⁻¹"
                    inline_unit = ""
                    value_start = 1
                    if len(cells) >= 3:
                        potential_unit = cells[1]
                        is_unit = (
                            len(potential_unit) <= 5
                            and not any(c.isdigit() for c in potential_unit)
                            and potential_unit not in ['-', '–', '—', '']
                        )
                        if is_unit:
                            inline_unit = potential_unit
                            value_start = 2  # La valeur est en cells[2]

                    for i, cell in enumerate(cells[value_start:], start=value_start):
                        if not cell or cell in ['-', '–', '—', '']:
                            continue
                        is_short_numeric = len(cell) <= 5 and any(c.isdigit() for c in cell)
                        if not is_short_numeric:
                            continue

                        unit = inline_unit or (headers[i] if i < len(headers) else "")
                        phrase = f"{label} : {cell} {unit}".strip()
                        sentences.append(phrase)
                        break

                return "\n".join(sentences)
            except Exception:
                return ""
    
        nonlocal current_table, table_intro
        if len(current_table) < 3: # Pas un vrai tableau Markdown
            if table_intro: # On restaure l'intro si ce n'était pas un vrai tableau
                current_text_block.insert(0, table_intro)
                table_intro = ""
            current_text_block.extend(current_table)
            current_table = []
            return
            
        hierarchy_prefix = get_hierarchy_context()
        
        # --- NOUVEAU : On fusionne l'intro avec le tableau ---
        intro_str = f"{table_intro}\n\n" if table_intro else ""
        
        header = current_table[0]
        separator = current_table[1]
        raw_data_rows = current_table[2:]
        
        
        data_rows = []
        for row in raw_data_rows:
            cells = [c.strip() for c in row.strip().strip('|').split('|')]
            first_cell = cells[0] if cells else ''
            is_continuation = first_cell in ('--', '–', '—', '') and data_rows
            if is_continuation:
                # On concatène le contenu à la dernière cellule Action de la ligne précédente
                prev_cells = [c.strip() for c in data_rows[-1].strip().strip('|').split('|')]
                # On ajoute le texte à la dernière cellule non vide
                continuation_text = ' '.join(c for c in cells if c and c not in ('--', '–', '—'))
                if continuation_text and prev_cells:
                    prev_cells[-1] = (prev_cells[-1] + ' / ' + continuation_text).strip(' /')
                    data_rows[-1] = '| ' + ' | '.join(prev_cells) + ' |'
            else:
                data_rows.append(row)
        
        # Contexte complet envoyé au LLM : Hiérarchie + Texte Introductif + Tableau complet
        full_table_markdown = hierarchy_prefix + intro_str + "\n".join(current_table)
        
        # Contexte découpé pour l'Embedding (Précision laser pour Jina)
        rows_per_batch = 5
        for i in range(0, len(data_rows), rows_per_batch):
            batch_rows = data_rows[i : i + rows_per_batch]
            mini_table = "\n".join([header, separator] + batch_rows)
            
            # Jina verra maintenant la phrase "Le tableau suivant montre..." !
            embedding_text = hierarchy_prefix + intro_str + mini_table
            prose = parse_table_to_prose(header, separator, batch_rows)
            if prose:
                embedding_text += "\n\n" + prose
            chunks.append({
                "text": embedding_text,      # Mini-tableau avec en-têtes + intro
                "llm_context": full_table_markdown,  # Tableau COMPLET + intro
                "source": source,
                "page": page_num,
                "is_table": True,
                "synthetic": False
            })
        current_table = []
        table_intro = "" # On réinitialise pour le prochain tableau

    # --- LECTURE LIGNE PAR LIGNE ---
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        is_table_line = line.strip().startswith('|') and line.strip().endswith('|')
        
        if header_match:
            flush_text()
            flush_table()
            
            level = len(header_match.group(1))
            # On efface les sous-titres plus profonds devenus obsolètes
            keys_to_remove = [k for k in active_headers.keys() if k >= level]
            for k in keys_to_remove:
                del active_headers[k]
                
            active_headers[level] = line.strip()
            # On ajoute quand même le titre dans le bloc de texte courant
            current_text_block.append(line) 
            
        elif is_table_line:
            if not in_table:
                # --- NOUVEAU : On ne flush plus le texte, on l'absorbe ! ---
                table_intro = "\n".join(current_text_block).strip()
                current_text_block = [] # On vide le bloc pour qu'il ne soit pas flushé en texte standard
                in_table = True
            current_table.append(line.strip())
            
        else:
            if in_table:
                flush_table() # Fin du tableau
                in_table = False
            
            if line.strip():
                current_text_block.append(line)
                
    # Vider les tampons à la fin de la page
    flush_text()
    flush_table()
    
    return chunks


def chunks_from_json(json_path: Path, tokenizer) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    stored_filepath = doc.get("filepath", "")
    try:
        source = str(Path(stored_filepath).relative_to("/app/data"))
    except ValueError:
        source = Path(stored_filepath).name

    all_chunks = []
    for page in doc.get("pages", []):
        raw_content = page.get("content", "")
        if not raw_content:
            continue

        content = ftfy.fix_text(raw_content)
        # On passe directement la page à notre processeur HiRAG
        
        content = re.sub(r'```[\w]*\s*```', '', content)        # blocs vides
        content = re.sub(r'```[\w]*\s*\n\s*```', '', content)   # blocs vides multi-ligne
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        page_chunks = process_page_hirag(content, source, page.get("page"), tokenizer)
        all_chunks.extend(page_chunks)

    return all_chunks


# ======================
# EMBEDDING
# ======================

def get_embeddings_gpu(model, texts: list[str], batch_size: int = 1) -> np.ndarray:
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i: i + batch_size]
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                batch_emb = model.encode_text(batch_texts, task="retrieval")

            if isinstance(batch_emb, torch.Tensor):
                batch_emb = batch_emb.detach().cpu().numpy()
            elif isinstance(batch_emb, list):
                batch_emb = np.array([
                    t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
                    for t in batch_emb
                ])

            all_embeddings.extend(batch_emb)

        except torch.OutOfMemoryError:
            print(f"  OOM au batch {i}. Nettoyage...")
            torch.cuda.empty_cache()
            gc.collect()
            raise

        if (i // batch_size + 1) % 50 == 0:
            print(f"  - Embeddé {min(i + batch_size, total)}/{total} chunks")

    return np.vstack(all_embeddings)


# ======================
# FAISS — LOAD / SAVE / MERGE
# ======================

def get_brand_paths(brand: str) -> tuple[str, str]:
    brand_dir  = os.path.join(EMBEDDINGS_ROOT, brand)
    faiss_path = os.path.join(brand_dir, f"{brand}.faiss")
    pkl_path   = os.path.join(brand_dir, f"{brand}.pkl")
    return faiss_path, pkl_path


def load_brand_index(brand: str) -> tuple[faiss.Index | None, list]:
    """Charge l'index FAISS et les chunks existants d'une marque."""
    faiss_path, pkl_path = get_brand_paths(brand)

    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        return None, []

    index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def save_brand_index(brand: str, index: faiss.Index, chunks: list):
    """Sauvegarde l'index FAISS et les chunks d'une marque."""
    faiss_path, pkl_path = get_brand_paths(brand)
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

    faiss.write_index(index, faiss_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  → Sauvegardé : {faiss_path} ({index.ntotal} vecteurs total)")


def merge_into_brand(brand: str, new_chunks: list, new_embeddings: np.ndarray):
    """
    Charge l'index existant de la marque et y ajoute les nouveaux vecteurs.
    Crée l'index s'il n'existe pas encore.
    """
    existing_index, existing_chunks = load_brand_index(brand)

    # Normalisation L2 pour la similarité cosinus
    new_embeddings = new_embeddings.astype("float32")
    faiss.normalize_L2(new_embeddings)

    if existing_index is None:
        print(f"  Création d'un nouvel index pour {brand}...")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
    else:
        print(f"  Merge dans l'index existant de {brand} ({existing_index.ntotal} vecteurs)...")
        index = existing_index

    index.add(new_embeddings)
    merged_chunks = existing_chunks + new_chunks

    save_brand_index(brand, index, merged_chunks)
    return index.ntotal


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    print("=" * 60)
    print("  Embedding incrémental — démarrage")
    print(f"  Source  : {PARSED_ROOT}")
    print(f"  Sortie  : {EMBEDDINGS_ROOT}")
    print("=" * 60)

    # ── Chargement de l'état ──────────────────────────────────
    state        = load_state()
    brand_files  = collect_json_files()

    if not brand_files:
        print("Aucun fichier JSON trouvé. Lancez d'abord le parser.")
        exit(0)

    # ── Détection des nouveaux fichiers ──────────────────────
    to_process: list[tuple[str, Path]] = []

    for brand, paths in brand_files.items():
        for json_path in paths:
            rel_key   = str(json_path.relative_to(PARSED_ROOT))
            signature = json_signature(json_path)

            if state.get(rel_key) == signature:
                pass  # Déjà embeddé, on skip
            else:
                to_process.append((brand, json_path))

    already_done = sum(len(v) for v in brand_files.values()) - len(to_process)
    print(f"\n  JSONs trouvés     : {sum(len(v) for v in brand_files.values())}")
    print(f"  Déjà embeddés     : {already_done}")
    print(f"  À traiter         : {len(to_process)}")

    if not to_process:
        print("\nTout est à jour. Rien à faire.")
        exit(0)

    # ── Chargement du modèle ─────────────────────────────────
    print("\nChargement du tokenizer et du modèle d'embedding...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    emb_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    emb_model.eval()
    print("Modèle prêt.\n")

    # ── Regroupement par marque pour merger en une passe ─────
    brand_batches: dict[str, list[tuple[Path, list]]] = {}

    print("Chunkisation de tous les nouveaux fichiers...")
    for idx, (brand, json_path) in enumerate(to_process, 1):
        rel = json_path.relative_to(PARSED_ROOT)
        print(f"  [{idx}/{len(to_process)}] {rel}")
        try:
            chunks = chunks_from_json(json_path, tokenizer)
            if not chunks:
                print(f"    (aucun chunk extrait, fichier ignoré)")
                continue
            brand_batches.setdefault(brand, []).append((json_path, chunks))
            print(f"    → {len(chunks)} chunks")
        except Exception as e:
            print(f"    ✗ Erreur : {e}")

    # ── Embedding + Merge par marque ─────────────────────────
    for brand, file_chunk_pairs in brand_batches.items():
        print(f"\n{'─'*50}")
        print(f"  Marque : {brand}  ({len(file_chunk_pairs)} nouveau(x) fichier(s))")
        print(f"{'─'*50}")

        all_new_chunks: list[dict] = []
        for _, chunks in file_chunk_pairs:
            all_new_chunks.extend(chunks)

        print(f"  Calcul des embeddings pour {len(all_new_chunks)} chunks...")
        texts = [c["text"] for c in all_new_chunks]

        try:
            embeddings = get_embeddings_gpu(emb_model, texts, BATCH_SIZE)
        except Exception as e:
            print(f"  ✗ Échec embedding pour {brand} : {e}")
            continue

        total_vecs = merge_into_brand(brand, all_new_chunks, embeddings)
        print(f"  ✓ Index {brand} : {total_vecs} vecteurs au total")

        # Mise à jour de l'état uniquement si tout s'est bien passé
        for json_path, _ in file_chunk_pairs:
            rel_key             = str(json_path.relative_to(PARSED_ROOT))
            state[rel_key]      = json_signature(json_path)

        save_state(state)

    print("\n" + "=" * 60)
    print("  Embedding terminé.")
    print("=" * 60)


