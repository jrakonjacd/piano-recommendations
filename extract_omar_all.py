import os
import csv
import numpy as np
import torch
import librosa
from pathlib import Path
from omar_rq import get_model

#CONFIG
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID   = "mtg-upf/omar-rq-multifeature-25hz-fsq"
MODEL_ID2  = "mtg-upf/omar-rq-base"
INPUT_DIR  = Path("../Piano syllabus/audio")
OUTPUT_DIR = Path("./omar_base_outputs_all")
LAYER      = 6
CHUNK_SEC  = 30.0   
OVERLAP_S  = 1.0    
SAVE_SEQ   = False  
SKIP_EXIST = True  
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

def load_audio_16k_mono(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr

def chunk_indices(n_samples, sr, chunk_sec=30.0, overlap_sec=1.0):
    """Return (start, end) sample indices for chunks with overlap."""
    if chunk_sec <= 0:
        return [(0, n_samples)]
    win = int(round(chunk_sec * sr))
    hop = int(round(max(chunk_sec - overlap_sec, 0.001) * sr))
    if n_samples <= win:
        return [(0, n_samples)]
    idx, s = [], 0
    while s < n_samples:
        e = min(s + win, n_samples)
        idx.append((s, e))
        if e >= n_samples:
            break
        s += hop
    return idx

def pool_mean_std(x_t_c: np.ndarray) -> np.ndarray:
    """x_t_c: (T, C) -> concat(mean, std) -> (2C,)"""
    m = x_t_c.mean(axis=0)
    s = x_t_c.std(axis=0, ddof=0)
    return np.concatenate([m, s], axis=0)

def find_audio_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p

def safe_stem(p: Path) -> str:
    return p.stem.replace(" ", "_")

model = get_model(model_id=MODEL_ID2, device=DEVICE)
model.eval().to(DEVICE)
eps = float(getattr(model, "eps", 25.0))  # embeddings per second
print(f"Device: {DEVICE}")
print(f"Model eps (embeddings/second): {eps}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
manifest_path = OUTPUT_DIR / "manifest.csv"

# Collect files already recorded in manifest
processed_files = set()
write_header = not manifest_path.exists()
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as mf_read:
        reader = csv.DictReader(mf_read)
        for row in reader:
            processed_files.add(row["file"])

# Open manifest for appending
mf = open(manifest_path, "a", encoding="utf-8", newline="")
mw = csv.writer(mf)
if write_header:
    mw.writerow([
        "status","file","sample_rate","duration_s","eps","frames_T","dim_C",
        "layer","est_duration_s","pooled_path","sequence_path","timestamps_path"
    ])

files = list(find_audio_files(INPUT_DIR))
print(f"Found {len(files)} audio files under {INPUT_DIR}")

for fpath in files:
    fpath_str = str(fpath)

    #Skip if already logged in manifest (any status)
    if fpath_str in processed_files:
        print(f"[SKIP] already in manifest: {fpath.name}")
        continue

    rel_out_dir = OUTPUT_DIR / fpath.parent.relative_to(INPUT_DIR)
    rel_out_dir.mkdir(parents=True, exist_ok=True)
    base = safe_stem(fpath)
    pool_path = rel_out_dir / f"{base}_layer{LAYER}_pooled_meanstd.npy"
    seq_path  = rel_out_dir / f"{base}_layer{LAYER}_sequence.npy"
    ts_path   = rel_out_dir / f"{base}_layer{LAYER}_timestamps.npy"

    if SKIP_EXIST and pool_path.exists() and (not SAVE_SEQ or seq_path.exists()):
        mw.writerow([
            "skipped", fpath_str, "", "", f"{eps:.6f}", "", "", LAYER, "",
            str(pool_path), str(seq_path) if SAVE_SEQ else "", str(ts_path) if SAVE_SEQ else ""
        ])
        mf.flush()
        print(f"[SKIP] exists on disk: {fpath.name}")
        continue

    try:
        y, sr = load_audio_16k_mono(fpath_str, sr=16000)
        dur_s = len(y) / sr

        chunks = []
        for s, e in chunk_indices(len(y), sr, chunk_sec=CHUNK_SEC, overlap_sec=OVERLAP_S):
            x = torch.from_numpy(y[s:e]).unsqueeze(0).to(DEVICE)  #(1, T')
            with torch.no_grad():
                emb = model.extract_embeddings(x, layers=[LAYER])  #(L,B,T,C)
            chunks.append(emb)

        embeddings = torch.cat(chunks, dim=2) if len(chunks) else torch.empty(1, 1, 0, 0, device=DEVICE)
        L, B, T, C = embeddings.shape

        if T > 0 and (torch.isnan(embeddings).any() or torch.isinf(embeddings).any()):
            raise ValueError("NaN/Inf in embeddings")

        if T > 0:
            timestamps = torch.arange(T, device=embeddings.device) / eps 
            est_dur = timestamps[-1].item()
        else:
            timestamps = torch.empty(0, device=DEVICE)
            est_dur = 0.0

      
        emb_np = embeddings.squeeze(0).squeeze(0).detach().cpu().numpy()  #(T,C)
        if T > 0:
            pooled = pool_mean_std(emb_np)  #(2C,)
        else:
            pooled = np.empty((0,), dtype=np.float32)

    
        # pooled
        tmp_pool = str(pool_path) + ".tmp"
        with open(tmp_pool, "wb") as f:
            np.save(f, pooled)
        os.replace(tmp_pool, pool_path)

        seq_save = ""
        ts_save  = ""
        if SAVE_SEQ:
            tmp_seq = str(seq_path) + ".tmp"
            tmp_ts  = str(ts_path) + ".tmp"
            with open(tmp_seq, "wb") as f:
                np.save(f, emb_np)
            with open(tmp_ts, "wb") as f:
                np.save(f, timestamps.detach().cpu().numpy())
            os.replace(tmp_seq, seq_path)
            os.replace(tmp_ts,  ts_path)
            seq_save = str(seq_path)
            ts_save  = str(ts_path)

        mw.writerow([
            "ok", fpath_str, sr, f"{dur_s:.6f}", f"{eps:.6f}", T, C, LAYER,
            f"{est_dur:.6f}", str(pool_path), seq_save, ts_save
        ])
        mf.flush()

        print(f"[OK] {fpath.name} -> T={T}, C={C}, pooled-> {pool_path.name}")

    except Exception as e:
        mw.writerow([
            f"error:{type(e).__name__}", fpath_str, "", "", "", "", "", LAYER, "", "", "", ""
        ])
        mf.flush()
        print(f"[ERR] {fpath} :: {e}")

mf.close()
print(f"\nDone. Manifest saved to: {manifest_path.resolve()}")
