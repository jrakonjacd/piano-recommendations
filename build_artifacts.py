import os
import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import umap
except Exception:
    umap = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except Exception:
    TSNE = None
    PCA = None


def l2norm_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def load_vectors_from_folder(folder: Path):
    files = sorted([p for p in folder.glob("*.npy") if p.is_file()])
    ids, vecs = [], []
    for p in files:
        try:
            v = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"[warn] cannot load {p.name}: {e}")
            continue
        if v.ndim == 1:
            ids.append(p.stem)
            vecs.append(v.astype(np.float32))
        elif v.ndim == 2 and v.shape[0] == 1:
            ids.append(p.stem)
            vecs.append(v.squeeze(0).astype(np.float32))
        else:
            continue
    if not ids:
        return np.array([], dtype=str), np.zeros((0, 0), dtype=np.float32)
    X = np.stack(vecs, axis=0)
    return np.array(ids, dtype=str), X


def ensure_art_dir(root: Path, name: str) -> Path:
    out = (root / name)
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_umap(X: np.ndarray, n_neighbors=30, min_dist=0.1, random_state=42):
    if umap is None:
        print("[warn] umap-learn not installed; falling back to PCA for coords_umap.npy")
        if PCA is None:
            raise RuntimeError("Neither umap-learn nor scikit-learn PCA available.")
        return PCA(n_components=2, random_state=random_state).fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, metric="cosine"
    )
    return reducer.fit_transform(X)


def compute_tsne(X: np.ndarray, perplexity=30, learning_rate="auto", random_state=42):
    if TSNE is None:
        raise RuntimeError("scikit-learn not installed; cannot compute t-SNE.")
    X_in = X
    if PCA is not None and X.shape[1] > 64:
        X_in = PCA(n_components=min(50, X.shape[1]-1), random_state=random_state).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                init="pca", random_state=random_state)
    return tsne.fit_transform(X_in)


def save_npy_atomic(path: Path, arr: np.ndarray):
    arr = np.asarray(arr)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def build_one_model(emb_dir: Path, art_root: Path, name: str,
                    make_umap: bool, make_tsne: bool, overwrite: bool):
    print(f"\n=== [{name}] ===")
    ids, X = load_vectors_from_folder(emb_dir)
    if ids.size == 0:
        print(f"[skip] no per-track vectors found in {emb_dir}")
        return

    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        raise ValueError(f"Found non-finite values in {emb_dir} at indices {bad[0][:10]}...")
    D = X.shape[1]
    print(f"[info] loaded {len(ids)} vectors, dim={D} from {emb_dir}")

    X = l2norm_rows(X)

    out_dir = ensure_art_dir(art_root, name)
    ids_p    = out_dir / "ids.npy"
    vecs_p   = out_dir / "vecs.npy"
    umap_p   = out_dir / "coords_umap.npy"
    tsne_p   = out_dir / "coords_tsne.npy"

    if ids_p.exists() and vecs_p.exists() and not overwrite:
        print("[ok] ids.npy / vecs.npy already exist (use --overwrite to rewrite)")
    else:
        save_npy_atomic(ids_p, ids)
        save_npy_atomic(vecs_p, X.astype(np.float32))
        print(f"[write] {ids_p.name}, {vecs_p.name}")

    if make_umap:
        if umap_p.exists() and not overwrite:
            print("[ok] coords_umap.npy exists")
        else:
            print("[run] UMAP...")
            U = compute_umap(X)
            save_npy_atomic(umap_p, U.astype(np.float32))
            print(f"[write] {umap_p.name}")

    if make_tsne:
        if tsne_p.exists() and not overwrite:
            print("[ok] coords_tsne.npy exists")
        else:
            print("[run] t-SNE...")
            T = compute_tsne(X)
            save_npy_atomic(tsne_p, T.astype(np.float32))
            print(f"[write] {tsne_p.name}")

    print("[done] artifact files under", out_dir)


def guess_models(emb_root: Path):
    """
    If emb_root has *.npy files directly treat as single model named after folder
    """
    direct_npys = list(emb_root.glob("*.npy"))
    if direct_npys:
        return [(emb_root, emb_root.name)]
    models = []
    for sub in sorted([p for p in emb_root.iterdir() if p.is_dir()]):
        if any(sub.glob("*.npy")):
            models.append((sub, sub.name))
    return models


def main():
    ap = argparse.ArgumentParser(description="Build missing artifacts from embeddings.")
    ap.add_argument("--emb_root", type=Path, default=Path("embeddings"),
                    help="Root with per-track *.npy vectors (flat or per-model subfolders).")
    ap.add_argument("--art_root", type=Path, default=Path("artifacts/psyllabus"),
                    help="Destination artifacts root used by the Dash app.")
    ap.add_argument("--model", type=str, default=None,
                    help="Optional: only build this model folder name (subdir of emb_root).")
    ap.add_argument("--umap", action="store_true", default=True, help="Compute UMAP coords.")
    ap.add_argument("--no-umap", dest="umap", action="store_false")
    ap.add_argument("--tsne", action="store_true", default=False, help="Also compute t-SNE coords.")
    ap.add_argument("--overwrite", action="store_true", default=False, help="Rewrite existing files.")
    args = ap.parse_args()

    emb_root: Path = args.emb_root.resolve()
    art_root: Path = args.art_root.resolve()
    art_root.mkdir(parents=True, exist_ok=True)

    if not emb_root.exists():
        print(f"[err] embeddings root not found: {emb_root}")
        sys.exit(1)

    todo = guess_models(emb_root)
    if args.model:
        # filter to one subfolder
        todo = [(p, name) for (p, name) in todo if name == args.model or p.name == args.model]
        if not todo:
            print(f"[err] model '{args.model}' not found under {emb_root}")
            sys.exit(1)

    if not todo:
        print(f"[err] no candidate models under {emb_root}")
        sys.exit(1)

    print(f"[info] embeddings root: {emb_root}")
    print(f"[info] artifacts root : {art_root}")
    print(f"[info] models to build: {', '.join(name for _, name in todo)}")
    print(f"[info] compute UMAP: {args.umap} | compute t-SNE: {args.tsne} | overwrite: {args.overwrite}")

    for emb_dir, name in todo:
        build_one_model(
            emb_dir=emb_dir,
            art_root=art_root,
            name=name,
            make_umap=args.umap,
            make_tsne=args.tsne,
            overwrite=args.overwrite
        )

    print("\n[ok] All done.")


if __name__ == "__main__":
    main()
