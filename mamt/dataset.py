import os
import glob
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, Subset

class MAMT_Dataset(Dataset):
    def __init__(self, mel_dir: str, target_dir: str):
        mel_paths = glob.glob(os.path.join(mel_dir, "*_mel.npz"))
        target_paths = glob.glob(os.path.join(target_dir, "*_target.npz"))

        def parse_key(p, suffix):
            file_name = Path(p).stem
            if file_name.endswith(suffix):
                file_name = file_name[:-len(suffix)]
            name, idx = file_name.rsplit("_", 1)
            return (name, int(idx))

        mel_map = {parse_key(p, "_mel"): p for p in mel_paths}
        target_map = {parse_key(p, "_target"): p for p in target_paths}

        self.keys = sorted(set(mel_map.keys()) & set(target_map.keys()))
        self.mel_map = mel_map
        self.tgt_map = target_map

        print(f"[Dataset] usable pairs: {len(self.keys)}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        name, idx = self.keys[i]
        mel_path = self.mel_map[(name, idx)]
        target_path = self.tgt_map[(name, idx)]

        mel = np.load(mel_path)["mel"]
        target = np.load(target_path)
        tokens = target["tokens"].astype(np.int64)
        onset = target["onset"].astype(np.float32)
        offset = target["offset"].astype(np.float32)
        
        mel = mel.T.astype(np.float32)

        return mel, tokens, onset, offset
    
def split_by_audio(
    dataset,
    train_ratio = 0.85,
    val_ratio = 0.1,
    test_ratio = 0.05,
    seed = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "sum of ratios should be 1."

    audio_to_idx = defaultdict(list)
    for i, (name, idx) in enumerate(dataset.keys):
        audio_to_idx[name].append(i)

    audio_names = list(audio_to_idx.keys())
    n_audio = len(audio_names)

    if n_audio < 3:
        raise ValueError(f"needs more train datas: n_audio={n_audio}")

    rng = random.Random(seed)
    rng.shuffle(audio_names)

    n_train_audios = int(n_audio * train_ratio)
    n_val_audios = int(n_audio * val_ratio)
    n_test_audios = n_audio - n_train_audios - n_val_audios

    if n_train_audios == 0:
        n_train_audios = 1
    if n_val_audios == 0 and n_audio >= 2:
        n_val_audios = 1
    if n_train_audios + n_val_audios > n_audio - 1:
        n_val_audios = max(1, n_audio - n_train_audios - 1)
    n_test_audios = n_audio - n_train_audios - n_val_audios

    train_audio_names = audio_names[:n_train_audios]
    val_audio_names = audio_names[n_train_audios:n_train_audios + n_val_audios]
    test_audio_names = audio_names[n_train_audios + n_val_audios:]

    train_indices = [i for name in train_audio_names for i in audio_to_idx[name]]
    val_indices = [i for name in val_audio_names   for i in audio_to_idx[name]]
    test_indices = [i for name in test_audio_names  for i in audio_to_idx[name]]

    print(f"[SPLIT]")
    print(f"  audio total {n_audio}: "
          f"train {len(train_audio_names)}, val {len(val_audio_names)}, test {len(test_audio_names)}")
    print(f"  segments total {len(dataset)}: "
          f"train {len(train_indices)}, val {len(val_indices)}, test {len(test_indices)}")

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    return train_ds, val_ds, test_ds