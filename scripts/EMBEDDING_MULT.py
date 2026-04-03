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
BATCH_SIZE     = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# STATE (idempotence)
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
# CHUNKING & TABLEAUX (100% GÉNÉRIQUE)
# ======================

def split_into_markdown(text: str, tokenizer, max_tokens: int, overlap: int = 100) -> list[str]:
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


def process_single_table(table_lines: list[str], context_lines: list[str], rows_per_chunk: int = 5) -> list[tuple[str, str]]:
    """
    Traitement 100% générique : découpe le tableau en petits blocs Markdown 
    auto-portants (avec en-tête) pour préserver la sémantique native.
    """
    if len(table_lines) < 3: 
        return []
    
    # On récupère le contexte factuel (ex: les 3 phrases avant le tableau)
    context_str = " ".join(context_lines[-3:]) if context_lines else ""
    
    # Le tableau complet original (pour que Qwen puisse tout lire d'un coup)
    full_table_markdown = "\n".join(context_lines[-2:] + table_lines)
    
    # Extraction de la structure Markdown
    header_line = table_lines[0]
    separator_line = table_lines[1]
    
    # On isole les données en ignorant les bordures vides
    data_lines = [l for l in table_lines[2:] if l.strip() and not re.match(r"^\|[\s\-\|]+\|$", l)]
    
    results = []
    
    # On crée des "mini-tableaux" par lots pour éviter de dépasser la limite de tokens
    for i in range(0, len(data_lines), rows_per_chunk):
        batch_lines = data_lines[i : i + rows_per_chunk]
        mini_table = [header_line, separator_line] + batch_lines
        mini_table_str = "\n".join(mini_table)
        
        # Ce que Jina va embedder : juste le contexte réel suivi du tableau brut.
        embedding_text = f"{context_str}\n\n{mini_table_str}".strip()
        
        results.append((embedding_text, full_table_markdown))
    
    return results


def extract_tables_from_page(page_text: str) -> tuple[list[tuple[str, str]], str]:
    """
    Parcourt le texte de la page JSON, isole les tableaux et leur contexte.

    Retourne :
        table_chunks : liste de (embedding_text, full_table_markdown)
        text_only    : le texte original dont tous les blocs tableau ont ete supprimes,
                       pret a etre passe a split_into_markdown sans duplication.
    """
    lines = [l.strip() for l in page_text.split("\n")]

    current_table   = []
    context_lines   = []
    in_table        = False
    table_chunks    = []
    text_only_lines = []   # lignes hors-tableau uniquement

    for l in lines:
        is_table_line = l.startswith("|") and l.endswith("|")

        if is_table_line:
            in_table = True
            current_table.append(l)
            # Ne PAS ajouter a text_only_lines -> c'est le fix du double-chunking
        else:
            if in_table:
                # Fin du bloc tableau -> traitement Parent-Child
                table_chunks.extend(process_single_table(current_table, context_lines))
                current_table = []
                in_table = False

            # Ligne de texte normal -> conservee dans le texte epure
            text_only_lines.append(l)

            clean_line = l.replace("#", "").replace("*", "").strip()
            if clean_line:
                context_lines.append(clean_line)
                if len(context_lines) > 3:
                    context_lines.pop(0)

    # Securite : page qui se termine directement sur un tableau
    if in_table and current_table:
        table_chunks.extend(process_single_table(current_table, context_lines))

    text_only = "\n".join(text_only_lines)
    return table_chunks, text_only


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

        # 1. Extraction des tableaux (Parent-Child) + recuperation du texte purgé
        #    extract_tables_from_page retourne desormais un tuple (chunks, text_sans_tableaux)
        table_data, text_only = extract_tables_from_page(content)

        for embedding_text, full_table in table_data:
            all_chunks.append({
                "text":        embedding_text,  # Mini-tableau + contexte (Lu par Jina)
                "llm_context": full_table,      # Tableau complet (Lu par Qwen)
                "source":      source,
                "page":        page.get("page"),
                "is_table":    True,
                "synthetic":   False,
            })

        # 2. Découpage classique sur le texte SANS les tableaux (fix du double-chunking)
        #    On passe text_only et non plus content pour eviter toute duplication
        chunks = split_into_markdown(text_only, tokenizer, MAX_TOKENS, overlap=CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append({
                "text":        chunk,
                "llm_context": chunk,
                "source":      source,
                "page":        page.get("page"),
                "is_table":    False,   # text_only ne contient plus de lignes | ... |
            })

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
