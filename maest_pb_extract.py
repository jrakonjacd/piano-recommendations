# maest_batch_extract.py
from pathlib import Path
import numpy as np, os
from tqdm import tqdm
from essentia.standard import MonoLoader, TensorflowPredictMAEST

AUDIO_DIR = Path("/mnt/c/Users/jrakonjac/OneDrive - Indra/Escritorio/Piano syllabus/audio")  # your MP3 root
PB        = "discogs-maest-30s-pw-2.pb"
OUT_CLS   = Path("embeddings/maest30_pw_l7_cls")              # 768-D
OUT_3TOK  = Path("embeddings/maest30_pw_l7_cls_dist_mean")    # 2304-D
OUT_CLS.mkdir(parents=True, exist_ok=True)
OUT_3TOK.mkdir(parents=True, exist_ok=True)

pred = TensorflowPredictMAEST(
    graphFilename=PB,
    output="PartitionedCall/Identity_7",
    patchSize=1876,     # OK for 30s model
    patchHopSize=1876,  # no overlap; set smaller for overlap if you want
    batchSize=1
)

def save_vec(path, v):
    v = v.astype(np.float32)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, v)
    os.replace(tmp, path)

def process_one(wav_or_mp3: Path):
    y = MonoLoader(filename=str(wav_or_mp3), sampleRate=16000, resampleQuality=4)()
    E = np.asarray(pred(y))           # (1,1,T,768)
    CLS  = E[0,0,0,:]
    DIST = E[0,0,1,:]
    AVG  = E[0,0,2:,:].mean(axis=0) if E.shape[2] > 2 else CLS  # guard

    # L2-normalize here so downstream stays consistent
    def l2(x): 
        n = np.linalg.norm(x); 
        return x if n==0 else x/n

    cls_vec  = l2(CLS)
    tri_vec  = l2(np.concatenate([CLS, DIST, AVG]))

    stem = wav_or_mp3.stem
    save_vec(OUT_CLS / f"{stem}.npy", cls_vec)     # 768
    save_vec(OUT_3TOK / f"{stem}.npy", tri_vec)    # 2304

if __name__ == "__main__":
    files = [p for p in AUDIO_DIR.rglob("*") if p.suffix.lower() in {".mp3",".wav",".flac",".m4a"}]
    for p in tqdm(files):
        try:
            process_one(p)
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")
