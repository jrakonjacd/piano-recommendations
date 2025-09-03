import os, pickle, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from model import AudioModel, ordinal_loss
import torch.serialization

PR_DIR  = Path("../Piano syllabus/pianoroll5")
CQT_DIR = Path("../Piano syllabus/cqt5")

MODEL_NAME = "audio_midi_multi_ps_v5"       
CKPT_PATH  = Path(f"models/{MODEL_NAME}/checkpoint_0.pth")

OUT_PRE  = Path(f"embeddings/{MODEL_NAME}_before_attention")
OUT_POST = Path(f"embeddings/{MODEL_NAME}_after_attention")
OUT_PRE.mkdir(parents=True, exist_ok=True)
OUT_POST.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu") 
torch.serialization.add_safe_globals({'ordinal_loss': ordinal_loss})
torch.set_grad_enabled(False)


MIN_T = 64
SHORT_RATIO = 0.5  # if onset_T < 0.5 * frames_T, synthesize onsets

def load_bin(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def to_TW_88(arr: np.ndarray) -> np.ndarray:
    """Return (T, 88). Accepts (T,88) or (88,T)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    if arr.shape[1] == 88:  # (T, 88)
        return arr.astype(np.float32)
    if arr.shape[0] == 88:  # (88, T) -> (T, 88)
        return arr.T.astype(np.float32)
    raise ValueError(f"Neither dimension is 88: {arr.shape}")

def synth_onsets(fr_TW: np.ndarray) -> np.ndarray:
    """Positive temporal derivative as onset proxy. fr_TW = (T,88)."""
    diff = np.diff(fr_TW, axis=0, prepend=fr_TW[:1, :])
    return np.maximum(diff, 0.0).astype(np.float32)

def build_pr_input(frames_path: Path, onset_path: Path) -> np.ndarray:
    """(C=2, T>=MIN_T, 88)"""
    fr = to_TW_88(load_bin(frames_path))
    if onset_path.exists():
        on = to_TW_88(load_bin(onset_path))
    else:
        on = synth_onsets(fr)

    if on.shape[0] < MIN_T or on.shape[0] < SHORT_RATIO * fr.shape[0]:
        on = synth_onsets(fr)

    T = min(fr.shape[0], on.shape[0])
    fr, on = fr[:T, :], on[:T, :]

    if T < MIN_T:
        pad = MIN_T - T
        fr = np.pad(fr, ((0, pad), (0, 0)), mode="edge")
        on = np.pad(on, ((0, pad), (0, 0)), mode="edge")

    pr_ct = np.stack([on, fr], axis=0)  # (2,T,88)
    return pr_ct.astype(np.float32)

def build_cqt_input(cqt_path: Path, target_T: int | None = None) -> np.ndarray:
    """(C=1, T>=MIN_T, 88). Optionally crop/pad to target_T."""
    cqt = to_TW_88(load_bin(cqt_path))  # (T,88)
    T = cqt.shape[0]

    if target_T is not None:
        # crop or pad to match PR time
        if T >= target_T:
            cqt = cqt[:target_T, :]
        else:
            pad = target_T - T
            cqt = np.pad(cqt, ((0, pad), (0, 0)), mode="edge")
    else:
        if T < MIN_T:
            pad = MIN_T - T
            cqt = np.pad(cqt, ((0, pad), (0, 0)), mode="edge")

    return cqt[np.newaxis, ...].astype(np.float32)  # (1,T,88)

def extract_and_save(stem: str, x_pr: torch.Tensor, x_cqt: torch.Tensor, model: AudioModel):
    """
    x_pr  : [1, 2, T, 88]
    x_cqt : [1, 1, T, 88]
    Saves: pre (512,) and post (256,)
    """
  
    before, after = model.extract_dual_embeddings_multi(x_pr, x_cqt)

    np.save(OUT_PRE  / f"{stem}_pre.npy",  before)
    np.save(OUT_POST / f"{stem}_post.npy", after)

def load_model() -> AudioModel:
    model = AudioModel(num_classes=11, rep="multi5", modality_dropout=False).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    # strict=True for the plain multi difficulty model
    _ = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model

def main(limit: int | None = None):
    model = load_model()

    pr_files = [p for p in PR_DIR.glob("*.bin") if not p.name.endswith("_onset.bin")]
    pr_stems = {p.stem for p in pr_files}
    cqt_stems = {p.stem for p in CQT_DIR.glob("*.bin")}
    stems = sorted(pr_stems & cqt_stems)
    if limit is not None:
        stems = stems[:limit]

    fail_log = OUT_POST.parent / f"{MODEL_NAME}_failures.txt"
    processed = skipped = failed = 0

    with open(fail_log, "a", encoding="utf-8") as flog:
        for stem in tqdm(stems):
            pre_out  = OUT_PRE  / f"{stem}_pre.npy"
            post_out = OUT_POST / f"{stem}_post.npy"
            if pre_out.exists() and post_out.exists():
                skipped += 1
                continue

            try:
                # Build PR (2,T,88)
                fr_path = PR_DIR / f"{stem}.bin"
                on_path = PR_DIR / f"{stem}_onset.bin"
                pr = build_pr_input(fr_path, on_path)           # (2,Tp,88)

                # Build CQT (1,T,88) aligned to PR time
                cqt = build_cqt_input(CQT_DIR / f"{stem}.bin", target_T=pr.shape[1])  # (1,Tp,88)

                # -> tensors with expected layout
                x_pr  = torch.tensor(pr,  dtype=torch.float32).unsqueeze(0).to(DEVICE)   # [1,2,T,88]
                x_cqt = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0).to(DEVICE)   # [1,1,T,88]

                extract_and_save(stem, x_pr, x_cqt, model)
                processed += 1

            except Exception as e:
                flog.write(f"{stem}\t{repr(e)}\n")
                flog.flush()
                failed += 1

    print(f"Done. processed={processed}, skipped={skipped}, failed={failed}, log={fail_log}")

if __name__ == "__main__":
    main(limit=None)
