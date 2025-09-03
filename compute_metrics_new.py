import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.stats import kendalltau

def npy(p: Path): return np.load(p, allow_pickle=True)

def l2norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def normalize_key(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    for t in ["_", ".", "-", "/", ":", ";"]:
        s = s.replace(t, " ")
    s = " ".join(s.split())
    return "".join(ch for ch in s if ch.isalnum() or ch == " ")

def to_int_or_none(x):
    try:
        return int(x)
    except Exception:
        return None

def load_global_meta(meta_json: Path):
    with open(meta_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    by_norm = {normalize_key(k): v for k, v in raw.items()}
    return raw, by_norm

def load_folder_meta(folder: Path):
    p = folder / "meta.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)  # id -> fields
        except Exception:
            pass
    return {}

def meta_for_id(pid: str, folder_meta, global_raw, global_norm):
   
    if pid in folder_meta:
        return folder_meta[pid]
   
    if pid in global_raw:
        return global_raw[pid]
   
    nid = normalize_key(pid)
    if nid in global_norm:
        return global_norm[nid]
   
    for nk, v in global_norm.items():
        if nk and nk in nid:
            return v
    return {}

def coverage_report(labels):
    """returns (n_total, n_labeled, n_classes_ge2)"""
    labels = np.array(labels, dtype=object)
    ok = [i for i, v in enumerate(labels) if v not in (None, "", np.nan)]
    n_total = len(labels)
    n_lab = len(ok)
    if n_lab == 0:
        return n_total, 0, 0
    vals = pd.Series([labels[i] for i in ok])
    cls_counts = vals.value_counts()
    n_ge2 = int((cls_counts >= 2).sum())
    return n_total, n_lab, n_ge2


def silhouette_cosine(X, y):
    return silhouette_score(X, y, metric="cosine")

def intra_inter_ratio(X, y, max_pairs=100000):
    """
    Average intra-class cosine distance divided by average inter-class cosine distance.
    Sampling is used if too many pairs.
    """
    N = len(y)
    D = 1 - (X @ X.T) 
   
    from collections import defaultdict
    idxs = defaultdict(list)
    for i, lab in enumerate(y):
        idxs[lab].append(i)

    intra_vals = []
    for lab, inds in idxs.items():
        if len(inds) < 2:
            continue
        inds = np.array(inds)
        W = D[np.ix_(inds, inds)]
        iu = np.triu_indices_from(W, k=1)
        vals = W[iu]
        if len(vals) > 0:
            intra_vals.append(vals)
    if not intra_vals:
        return np.nan
    intra_all = np.concatenate(intra_vals)
    labs = list(idxs.keys())
    inter_list = []
    for i in range(len(labs)):
        for j in range(i+1, len(labs)):
            a = np.array(idxs[labs[i]])
            b = np.array(idxs[labs[j]])
            block = D[np.ix_(a, b)].ravel()
            inter_list.append(block)
    if not inter_list:
        return np.nan
    inter_all = np.concatenate(inter_list)
    if inter_all.size > max_pairs:
        rng = np.random.default_rng(42)
        inter_all = rng.choice(inter_all, size=max_pairs, replace=False)
    return float(np.mean(intra_all) / (np.mean(inter_all) + 1e-12))

def precision_at_k(vecs, labels, ks=(5,10,20)):
    """
    Mean fraction of top-k neighbors (excluding self) that share the same label.
    """
    labels = np.array(labels, dtype=object)
    valid = np.array([lab not in (None, "", np.nan) for lab in labels], dtype=bool)
    if valid.sum() < 2:
        return {k: np.nan for k in ks}
    X = vecs
    sims = X @ X.T  # cosine since L2
    np.fill_diagonal(sims, -1e9)
    results = {k: [] for k in ks}
    for i in np.where(valid)[0]:
        lab = labels[i]
        s = sims[i]
        order = np.argpartition(-s, range(max(ks)))[:max(ks)]
        order = order[np.argsort(-s[order])]
        same = (labels[order] == lab)
        for k in ks:
            results[k].append(np.mean(same[:k]))
    return {k: float(np.mean(v)) if len(v) else np.nan for k, v in results.items()}

def kendall_tau_vs_distance(vecs, difficulties, max_pairs=200000):
    """
    Kendall's tau between pairwise cosine distance and |diff_i - diff_j|.
    """
    d = np.array([to_int_or_none(x) for x in difficulties])
    ok = np.where(~pd.isna(d))[0]
    if ok.size < 3:
        return np.nan, np.nan
    X = vecs[ok]
    diffs = d[ok]
    D = 1 - (X @ X.T)
    iu = np.triu_indices_from(D, k=1)
    dist_vec = D[iu]
    diff_vec = np.abs(diffs[:,None] - diffs[None,:])[iu]
    if dist_vec.size > max_pairs:
        rng = np.random.default_rng(42)
        sel = rng.choice(dist_vec.size, size=max_pairs, replace=False)
        dist_vec = dist_vec[sel]
        diff_vec = diff_vec[sel]
    tau, p = kendalltau(dist_vec, diff_vec)
    return float(tau), float(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_root", type=Path, default=Path("artifacts/psyllabus"))
    ap.add_argument("--meta_json", type=Path, required=True,
                    help="Global metadata JSON with youtube_link, composer, era, ps_rating, etc.")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[5,10,20])
    args = ap.parse_args()

    art_root = args.art_root.resolve()
    if not art_root.exists():
        raise SystemExit(f"Artifacts root not found: {art_root}")

    out_dir = args.out_dir or (art_root / "_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    global_raw, global_norm = load_global_meta(args.meta_json)

    models = []
    for d in sorted(art_root.iterdir()):
        if not d.is_dir(): continue
        if (d/"ids.npy").exists() and (d/"vecs.npy").exists():
            models.append(d)

    rows_lbl = []
    rows_tau = []

    print("\n=== Label coverage & metrics ===")
    for d in models:
        ids = npy(d/"ids.npy").astype(str)
        X   = l2norm_rows(npy(d/"vecs.npy").astype(np.float32))
        folder_meta = load_folder_meta(d)

        composers, eras, diffs = [], [], []
        for pid in ids:
            m = meta_for_id(pid, folder_meta, global_raw, global_norm)
            composers.append(m.get("composer") or None)
            eras.append(m.get("period") or m.get("era") or None)
            diffs.append(m.get("ps_rating") or m.get("ps") or m.get("difficulty"))

        for field, labels in [("composer", composers), ("era", eras), ("difficulty", diffs)]:
            n_total, n_lab, n_ge2 = coverage_report(labels)
            coverage_note = f"coverage: {n_lab}/{n_total}, classes>=2: {n_ge2}"
            if n_lab < 3 or n_ge2 < 2:
                rows_lbl.append(dict(
                    model=d.name, label_field=field,
                    silhouette_cosine=np.nan,
                    intra_inter_ratio=np.nan,
                    **{f"P@{k}": np.nan for k in args.ks},
                    _coverage=coverage_note
                ))
                continue

            mask = np.array([lab not in (None, "", np.nan) for lab in labels], dtype=bool)
            X_ = X[mask]
            y  = np.array(labels, dtype=object)[mask]

            # metrics
            try:
                sil = silhouette_cosine(X_, y)
            except Exception:
                sil = np.nan
            try:
                ratio = intra_inter_ratio(X_, y)
            except Exception:
                ratio = np.nan
            try:
                pk = precision_at_k(X, labels, ks=tuple(args.ks))
            except Exception:
                pk = {k: np.nan for k in args.ks}

            row = dict(model=d.name, label_field=field,
                       silhouette_cosine=sil, intra_inter_ratio=ratio,
                       _coverage=coverage_note)
            for k in args.ks:
                row[f"P@{k}"] = pk.get(k, np.nan)
            rows_lbl.append(row)

        # Kendall's tau (difficulty only)
        tau, p = kendall_tau_vs_distance(X, diffs)
        rows_tau.append(dict(model=d.name,
                             kendall_tau_diff_vs_distance=tau,
                             kendall_p_value=p))

    df_lbl = pd.DataFrame(rows_lbl)
    df_tau = pd.DataFrame(rows_tau)

    pd.set_option("display.max_rows", 200)
    print("\n=== Label-based metrics (silhouette, intra/inter, P@k) ===")
    cols = ["model", "label_field", "silhouette_cosine", "intra_inter_ratio"] + [f"P@{k}" for k in args.ks]
    if "_coverage" in df_lbl.columns:
        cols.append("_coverage")
    print(df_lbl[cols].to_string(index=False))

    print("\n=== Difficulty (ordinal) metrics (Kendall's tau) ===")
    print(df_tau.to_string(index=False))

    out_lbl = out_dir / "label_metrics.csv"
    out_tau = out_dir / "ordinal_metrics.csv"
    df_lbl.to_csv(out_lbl, index=False)
    df_tau.to_csv(out_tau, index=False)
    print("\nSaved:")
    print(f"  {out_lbl}")
    print(f"  {out_tau}")

if __name__ == "__main__":
    main()
