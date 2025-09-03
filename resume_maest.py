from pathlib import Path
import numpy as np, os, sys, argparse
from tqdm import tqdm
from essentia.standard import MonoLoader, TensorflowPredictMAEST

ap = argparse.ArgumentParser()
ap.add_argument("--audio_dir", type=Path, required=True,
                help="Root folder with mp3/wav/flac/m4a")
ap.add_argument("--pb", type=str, default="discogs-maest-30s-pw-2.pb",
                help="Path to MAEST .pb graph")
ap.add_argument("--out_cls", type=Path, default=Path("embeddings/maest30_pw_l7_cls"),
                help="Output dir for CLS (768-D)")
ap.add_argument("--out_tritok", type=Path, default=Path("embeddings/maest30_pw_l7_cls_dist_mean"),
                help="Output dir for [CLS|DIST|AVG] (2304-D)")
ap.add_argument("--overwrite", action="store_true",
                help="Recompute even if outputs already exist")
ap.add_argument("--limit", type=int, default=None,
                help="Optional cap on number of files to process")
ap.add_argument("--exts", nargs="+",
                default=[".mp3",".wav",".flac",".m4a"],
                help="Audio extensions to include")
args = ap.parse_args()

AUDIO_DIR = args.audio_dir
PB        = args.pb
OUT_CLS   = args.out_cls
OUT_3TOK  = args.out_tritok
OUT_CLS.mkdir(parents=True, exist_ok=True)
OUT_3TOK.mkdir(parents=True, exist_ok=True)
EXTS = {e.lower() for e in args.exts}

pred = TensorflowPredictMAEST(
    graphFilename=PB,
    output="PartitionedCall/Identity_7",  # layer7 embeddings
    patchSize=1876,     # 30s model expects 1876 x 96 mel patches
    patchHopSize=1876, 
    batchSize=1
)

def relkey(path: Path) -> str:
    """Unique, file-system-safe key derived from relative path (without extension)."""
    rel = path.relative_to(AUDIO_DIR).with_suffix("") 
    return str(rel).replace("/", "__").replace("\\", "__")

def save_vec(path: Path, v: np.ndarray):
    v = v.astype(np.float32, copy=False)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, v)
    os.replace(tmp, path)

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else (x / n)

def outputs_for(key: str):
    return OUT_CLS / f"{key}.npy", OUT_3TOK / f"{key}.npy"

def have_both(key: str) -> bool:
    p_cls, p_tri = outputs_for(key)
    return p_cls.exists() and p_tri.exists()

def need_any(key: str) -> tuple[bool, bool]:
    """Return (need_cls, need_tri) based on existing files and overwrite flag."""
    p_cls, p_tri = outputs_for(key)
    if args.overwrite:
        return True, True
    return (not p_cls.exists()), (not p_tri.exists())

files = [p for p in AUDIO_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
files.sort()
if args.limit:
    files = files[:args.limit]

processed = skipped = failed = 0

for f in tqdm(files, desc="MAEST extract"):
    key = relkey(f)
    need_cls, need_tri = need_any(key)

    if not (need_cls or need_tri):
        skipped += 1
        continue

    try:
        # Load audio 16k mono
        y = MonoLoader(filename=str(f), sampleRate=16000, resampleQuality=4)()

        E = np.asarray(pred(y))
        if E.ndim != 4 or E.shape[1] != 1 or E.shape[-1] != 768:
            raise RuntimeError(f"Unexpected MAEST output shape {E.shape}")

        CLS  = E[0, 0, 0, :]
        DIST = E[0, 0, 1, :] if E.shape[2] > 1 else CLS
        AVG  = E[0, 0, 2:, :].mean(axis=0) if E.shape[2] > 2 else CLS

        if need_cls:
            save_vec(OUT_CLS / f"{key}.npy", l2norm(CLS))

        if need_tri:
            tri = np.concatenate([CLS, DIST, AVG], axis=0)
            save_vec(OUT_3TOK / f"{key}.npy", l2norm(tri))

        processed += 1

    except Exception as e:
        failed += 1
        print(f"[ERR] {f}: {e}", file=sys.stderr)

print(f"\nDone. processed={processed}, skipped={skipped}, failed={failed}")
print(f"CLS dir:   {OUT_CLS.resolve()}")
print(f"3TOK dir:  {OUT_3TOK.resolve()}")
