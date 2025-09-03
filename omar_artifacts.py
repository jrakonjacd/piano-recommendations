import numpy as np, pathlib as p

SRC = p.Path("omar_outputs_all")  
DST = p.Path("artifacts/psyllabus/omar_rq_25hz_layer6_meanstd")
DST.mkdir(parents=True, exist_ok=True)

pairs = []
for fp in SRC.rglob("*_layer6_pooled_meanstd.npy"):
    pairs.append((fp, np.load(fp)))

ids  = np.array([fp.stem.replace("_layer6_pooled_meanstd","") for fp,_ in pairs])
vecs = np.vstack([v for _,v in pairs]).astype("float32")

np.save(DST/"ids.npy",  ids)
np.save(DST/"vecs.npy", vecs)
print("Saved:", DST/"ids.npy", DST/"vecs.npy", vecs.shape)
