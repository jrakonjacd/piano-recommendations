# pr_extraction.py
import os
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from model import AudioModel, ordinal_loss
import torch.serialization

"""
Pianoroll embeddings extractor (PR baseline).
- Input:   pairs "<name>.bin" (frames) and "<name>_onset.bin" (onsets)
- Output:  one file per piece in:
              embeddings/<model_name>_before_attention/<name>_pre.npy   # (512,)
              embeddings/<model_name>_after_attention/<name>_post.npy    # (256,)
- Robust:  handles missing/short/mismatched onset files and pads time to avoid pooling collapse.
"""

# =========================
# CONFIG
# =========================
# TIP: for faster I/O on WSL, copy bins to a Linux path (e.g., /home/<user>/data/pianoroll5)
PR_BIN_DIR = Path("/mnt/c/Users/jrakonjac/OneDrive - Indra/Escritorio/Piano syllabus/pianoroll5")

MODEL_NAME = "audio_midi_pr5era_v1"  # PR baseline (difficulty-only)
# MODEL_NAME = "audio_midi_pr5era_v1"       # PR + Era (aux head) -> will load with strict=False

CKPT_PATH = Path(f"models/{MODEL_NAME}/checkpoint_0.pth")

OUT_DIR_PRE  = Path(f"embeddings/{MODEL_NAME}_before_attention")
OUT_DIR_POST = Path(f"embeddings/{MODEL_NAME}_after_attention")
OUT_DIR_PRE.mkdir(parents=True, exist_ok=True)
OUT_DIR_POST.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")  # keep on CPU for deterministic extraction
torch.serialization.add_safe_globals({'ordinal_loss': ordinal_loss})
torch.set_grad_enabled(False)

# Minimum time length after alignment to survive 3 pools (3,4,4) @ 5 fps
MIN_T = 64          # safe (48 is the theoretical minimum)
SHORT_RATIO = 0.5   # if onset_T < SHORT_RATIO * frames_T, synthesize onsets


# =========================
# HELPERS
# =========================
def load_bin(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def to_TF(arr: np.ndarray) -> np.ndarray:
    """Normalize 2D array to (T, 88). Accepts (T,88) or (88,T)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    if arr.shape[1] == 88:  # (T,88)
        return arr.astype(np.float32)
    if arr.shape[0] == 88:  # (88,T) -> (T,88)
        return arr.T.astype(np.float32)
    raise ValueError(f"Neither dim is 88: {arr.shape}")

def synth_onsets(fr_TF: np.ndarray) -> np.ndarray:
    """Onset proxy = positive temporal derivative over time axis."""
    diff = np.diff(fr_TF, axis=0, prepend=fr_TF[:1, :])
    return np.maximum(diff, 0.0).astype(np.float32)

def build_pr_ct(frames_path: Path, onset_path: Path,
                min_T: int = MIN_T, short_ratio: float = SHORT_RATIO) -> np.ndarray:
    """
    Build a robust PR tensor (C=2, T, 88): channel 0 = onsets, channel 1 = frames.
      - If onset file is missing or much shorter than frames, synthesize onsets.
      - Align by cropping to the common min T.
      - If T < min_T, pad by repeating the last frame to avoid pooling collapse.
    """
    fr = to_TF(load_bin(frames_path))                 # (T_f, 88)
    if onset_path.exists():
        on = to_TF(load_bin(onset_path))             # (T_o, 88)
    else:
        on = synth_onsets(fr)

    # Fallback if onset stream is clearly too short
    if on.shape[0] < min_T or on.shape[0] < short_ratio * fr.shape[0]:
        on = synth_onsets(fr)

    # Align by cropping
    T = min(fr.shape[0], on.shape[0])
    fr = fr[:T, :]
    on = on[:T, :]

    # Pad time if needed
    if T < min_T:
        pad = min_T - T
        fr = np.pad(fr, ((0, pad), (0, 0)), mode="edge")
        on = np.pad(on, ((0, pad), (0, 0)), mode="edge")

    pr_ct = np.stack([on, fr], axis=0).astype(np.float32)  # (2, T>=min_T, 88)
    return pr_ct

def extract_and_save(stem: str, x_tensor: torch.Tensor, model: AudioModel):
    """
    x_tensor: [1, 2, T, 88]
    Saves:
      - OUT_DIR_PRE / f"{stem}_pre.npy"   -> (512,)
      - OUT_DIR_POST/ f"{stem}_post.npy"  -> (256,)
    """
    before, after = model.extract_dual_embeddings2(x_tensor)  # pre (mean+std), post (attn)
    np.save(OUT_DIR_PRE  / f"{stem}_pre.npy",  before)
    np.save(OUT_DIR_POST / f"{stem}_post.npy", after)


# =========================
# MODEL
# =========================
def load_model() -> AudioModel:
    model = AudioModel(num_classes=11, rep="pianoroll5", modality_dropout=False).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    # difficulty-only -> strict=True; era variant -> strict=False
    strict = not MODEL_NAME.endswith("era_v1")
    _ = model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    model.eval()
    return model


# =========================
# MAIN
# =========================
def main(limit: int | None = None):
    model = load_model()

    # List base files (exclude *_onset.bin)
    files = sorted([p for p in PR_BIN_DIR.glob("*.bin") if not p.name.endswith("_onset.bin")])
    if limit is not None:
        files = files[:limit]

    # Resumable + logging
    fail_log_path = OUT_DIR_POST.parent / f"{MODEL_NAME}_failures.txt"
    processed = 0
    skipped = 0
    failed = 0

    with open(fail_log_path, "a", encoding="utf-8") as flog:
        for f in tqdm(files):
            stem = f.stem
            pre_out  = OUT_DIR_PRE  / f"{stem}_pre.npy"
            post_out = OUT_DIR_POST / f"{stem}_post.npy"

            # Skip if already extracted
            if pre_out.exists() and post_out.exists():
                skipped += 1
                continue

            try:
                onset = f.with_name(f.stem + "_onset.bin")
                pr_ct = build_pr_ct(f, onset)  # (2, T, 88)

                x = torch.tensor(pr_ct, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,2,T,88]
                extract_and_save(stem, x, model)
                processed += 1

            except Exception as e:
                flog.write(f"{stem}\t{repr(e)}\n")
                flog.flush()
                failed += 1

    print(f"Done. processed={processed}, skipped={skipped}, failed={failed}, log={fail_log_path}")

if __name__ == "__main__":
    # Pass a small integer to 'limit' to test on a subset first (e.g., main(limit=200))
    main(limit=None)
