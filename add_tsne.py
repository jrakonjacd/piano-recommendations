import argparse, json, numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True, type=Path,
                    help="e.g. artifacts/psyllabus/audio_midi_pianoroll_ps_5_v4_post")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=None,
                    help="default: min(30, (N-1)/3)")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--n_iter", type=int, default=1500)
    ap.add_argument("--subset", type=int, default=None,
                    help="optional: subsample N for faster t-SNE")
    args = ap.parse_args()

    V = np.load(args.artifact_dir / "vecs_l2.npy")
    ids = np.load(args.artifact_dir / "ids.npy", allow_pickle=True)
    N = V.shape[0]

    if args.subset and args.subset < N:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(N, size=args.subset, replace=False)
        V = V[idx]
        ids = ids[idx]
        print(f"[tsne] subsetting to N={len(ids)}")

   
    pca_dim = min(args.pca_dim, V.shape[1])
    Vp = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(V).astype(np.float32)

    print("[tsne] computing cosine distance matrix …")
    D = pairwise_distances(Vp, metric="cosine") 
    D = D.astype(np.float32, copy=False)

    
    perp = args.perplexity or min(30.0, (Vp.shape[0]-1)/3.0)
    perp = max(5.0, perp)  
    print(f"[tsne] using perplexity={perp:.1f}, n_iter={args.n_iter}, seed={args.seed}")

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        learning_rate="auto",
        init="pca",
        perplexity=perp,
        n_iter=args.n_iter,
        random_state=args.seed,
        verbose=1,
        square_distances=True,
    )
    Z = tsne.fit_transform(D).astype(np.float32)

    out = args.artifact_dir / "tsne.npy"
    np.save(out, Z)
    print(f"[tsne] saved {out} with shape {Z.shape}")

    summ_path = args.artifact_dir / "summary.json"
    if summ_path.exists():
        try:
            s = json.loads((args.artifact_dir / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            s = {}
    else:
        s = {}
    s["tsne"] = {
        "metric": "cosine(precomputed)",
        "perplexity": float(perp),
        "pca_dim": int(pca_dim),
        "n_iter": int(args.n_iter),
        "seed": int(args.seed),
        "subset_N": int(Vp.shape[0]),
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("[tsne] summary.json updated")

if __name__ == "__main__":
    main()
