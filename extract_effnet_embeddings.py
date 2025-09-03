import os
import csv
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def pool_mean_std(M: np.ndarray) -> np.ndarray:
    m = M.mean(axis=0)
    s = M.std(axis=0, ddof=0)
    return np.concatenate([m, s], axis=0)

def find_audio_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p

def safe_stem(p: Path) -> str:
    return p.stem.replace(" ", "_")

def save_npy_atomic(path: Path, arr: np.ndarray):
    arr = arr.astype(np.float32, copy=False)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, type=Path)
    ap.add_argument("--pb", required=True, type=str,
                    help="discogs_track_embeddings-effnet-bs64-1.pb")
    ap.add_argument("--out_dir", default=Path("embeddings/effnet_track_l1280_meanstd"), type=Path)
    ap.add_argument("--manifest", default="embeddings/effnet_track_manifest.csv")
    ap.add_argument("--skip_exist", action="store_true", default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    write_header = not manifest_path.exists()
    processed_files = set()

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as mf:
            for row in csv.DictReader(mf):
                processed_files.add(row["file"])

    pred = TensorflowPredictEffnetDiscogs(
        graphFilename=args.pb,
        output="PartitionedCall:1" 
    )

    files = list(find_audio_files(args.audio_dir))
    print(f"Found {len(files)} audio files under {args.audio_dir}")

    with open(manifest_path, "a", encoding="utf-8", newline="") as mf:
        mw = csv.writer(mf)
        if write_header:
            mw.writerow(["status","file","duration_s","frames_T","dim_C","pooled_path"])

        for fpath in tqdm(files, ncols=100):
            fstr = str(fpath)
            stem = safe_stem(fpath)
            out_path = args.out_dir / f"{stem}.npy"

            if fstr in processed_files:
                continue
            if args.skip_exist and out_path.exists():
                mw.writerow(["skipped", fstr, "", "", "", str(out_path)])
                mf.flush()
                continue

            try:
                audio = MonoLoader(filename=fstr, sampleRate=16000, resampleQuality=4)()
                dur_s = len(audio) / 16000.0

                E = np.asarray(pred(audio)) 

                if E.ndim == 1:
                    E = E[None, :]          
                elif E.ndim == 3:
                    E = E.squeeze(0)           

                if E.ndim != 2:
                    raise ValueError(f"Unexpected embedding shape {E.shape}")

                T, C = E.shape
                pooled = pool_mean_std(E)
                pooled = l2norm(pooled)

                save_npy_atomic(out_path, pooled)
                mw.writerow(["ok", fstr, f"{dur_s:.6f}", T, C, str(out_path)])
                mf.flush()

            except Exception as e:
                mw.writerow([f"error:{type(e).__name__}", fstr, "", "", "", ""])
                mf.flush()
                print(f"[ERR] {fpath.name}: {e}")

    print(f"Done. Manifest -> {manifest_path.resolve()}")
    print(f"Embeddings -> {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()
