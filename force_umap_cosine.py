# force_umap_cosine.py
import numpy as np
from pathlib import Path
import umap

PARAMS = dict(n_neighbors=30, min_dist=0.10, metric="cosine", random_state=42)

DIRS = [
    "artifacts/psyllabus/audio_midi_pianoroll_ps_5_v4_post",
    "artifacts/psyllabus/omar_rq_post",
]  # add more

for d in DIRS:
    p = Path(d)
    X = np.load(p/"X.npy")
    reducer = umap.UMAP(**PARAMS)
    coords = reducer.fit_transform(X).astype("float32")
    np.save(p/"umap_2d.npy", coords)
    print(f"UMAP recomputed for {d}: {coords.shape}")
