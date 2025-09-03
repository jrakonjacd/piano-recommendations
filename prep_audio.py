import os, csv, soundfile as sf
import numpy as np
import librosa

IN_CSV = "data/tracks.csv"
OUT_DIR = "data/audio16k"
SR = 16000
PEAK_TARGET = 0.9

os.makedirs(OUT_DIR, exist_ok=True)

with open(IN_CSV) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        in_path = row["path"]
        y, sr = librosa.load(in_path, sr=SR, mono=True)  # mono+resample
        peak = np.max(np.abs(y)) + 1e-9
        y = y * (PEAK_TARGET / peak)                     # normalización peak
        out_path = os.path.join(OUT_DIR, os.path.basename(in_path).replace(".mp3",".wav"))
        sf.write(out_path, y, SR)
        print("prep:", in_path, "->", out_path)
