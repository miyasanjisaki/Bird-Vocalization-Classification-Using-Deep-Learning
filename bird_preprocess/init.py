import os
import glob
import random
import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
from librosa import feature
import soundfile as sf

import torch
from torch.utils.data import Dataset, DataLoader
import sys, importlib

# 删除可能被本地 init.py 或其他文件污染的模块引用
for key in list(sys.modules.keys()):
    if key.startswith("init") or key.startswith("feature"):
        del sys.modules[key]

# 重新导入 librosa
import librosa
librosa = importlib.import_module("librosa")

# -------------------------
# 1) 建立索引（folders -> CSV）
# -------------------------
def build_index(root_dir: str, out_csv: str, exts=('wav', 'flac', 'mp3')):
    """
    Scan subfolders under root_dir; each subfolder name is a class label.
    Save CSV with columns: path,label
    """
    rows = []
    for label_name in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        for ext in exts:
            pattern = os.path.join(label_dir, f'**/*.{ext}')
            for p in glob.glob(pattern, recursive=True):
                rows.append((p, label_name))
    df = pd.DataFrame(rows, columns=['path', 'label'])
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"Index saved to {out_csv}, total files: {len(df)}")
    return df

# -------------------------
# 2) 预处理与特征提取函数
# -------------------------
def load_audio(path: str,
               sr: int = 16000,
               mono: bool = True,
               offset: float = 0.0,
               duration: Optional[float] = None,
               max_len_sec: float = 1.0) -> Tuple[np.ndarray, int]:
    # 加载音频
    y, orig_sr = librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration)

    # 计算目标长度（样本点数）
    max_len = int(max_len_sec * sr)

    # 若音频过短，补零
    if len(y) < max_len:
        pad_len = max_len - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')

    # 若音频过长，截断
    elif len(y) > max_len:
        y = y[:max_len]

    return y, sr

def highpass_filter(y: np.ndarray, sr: int, cutoff=300.0, order=2):
    """
    Simple Butterworth highpass filter implemented via librosa's effects if scipy not available.
    If scipy available, better to use scipy.signal.butter + filtfilt.
    Here implement with a naive FFT-based highpass (simple & dependency-free).
    NOTE: for production use, replace with scipy.signal.butter + filtfilt for stable results.
    """
    # naive freq-domain highpass
    if cutoff <= 0:
        return y
    n = len(y)
    # next power of two for FFT speed
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    mask = freqs >= cutoff
    Y_filtered = Y * mask
    y_hp = np.fft.irfft(Y_filtered, n=n)
    return y_hp.astype(np.float32)

def normalize_audio(y: np.ndarray, eps=1e-6):
    """
    RMS normalization to -20 dBFS (or unit RMS).
    """
    rms = np.sqrt(np.mean(y**2))
    if rms < eps:
        return y
    return y / (rms + eps)

def trim_silence(y: np.ndarray, top_db=40):
    """
    Trim leading/trailing silence using librosa.effects.trim
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def compute_mel_spectrogram(y: np.ndarray,
                            sr: int = 16000,
                            n_mels: int = 128,
                            n_fft: int = 1024,
                            hop_length: int = 256,
                            power: int = 2):
    """
    Return mel spectrogram in dB: shape (n_mels, T)
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256)

    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

# -------------------------
# 3) 数据增强（可选）
# -------------------------
def add_white_noise(y: np.ndarray, snr_db: float = 20.0):
    """
    Add white noise to achieve a target SNR (dB)
    """
    rms_signal = np.sqrt(np.mean(y**2))
    rms_noise = rms_signal / (10**(snr_db/20.0))
    noise = np.random.normal(0, rms_noise, size=y.shape).astype(np.float32)
    return (y + noise).astype(np.float32)

def time_shift(y: np.ndarray, shift_max: float, sr: int):
    """
    Random time shift in seconds (positive shift inserts zeros at start)
    """
    shift = int(random.uniform(-shift_max, shift_max) * sr)
    if shift > 0:
        y_shift = np.concatenate([np.zeros(shift, dtype=y.dtype), y])[:len(y)]
    elif shift < 0:
        y_shift = np.concatenate([y[-shift:], np.zeros(-shift, dtype=y.dtype)])
    else:
        y_shift = y
    return y_shift

# Simple SpecAugment (time and freq mask applied on mel spectrogram)
def spec_augment(mel: np.ndarray,
                 time_mask_param=20,
                 freq_mask_param=10,
                 num_time_masks=1,
                 num_freq_masks=1):
    mel_aug = mel.copy()
    n_mels, T = mel_aug.shape
    # time masks
    for _ in range(num_time_masks):
        t = random.randrange(0, time_mask_param+1) if T > 0 else 0
        t0 = random.randrange(0, max(1, T - t + 1))
        mel_aug[:, t0:t0+t] = mel_aug.min()  # set to min dB (or 0)
    # freq masks
    for _ in range(num_freq_masks):
        f = random.randrange(0, freq_mask_param+1) if n_mels > 0 else 0
        f0 = random.randrange(0, max(1, n_mels - f + 1))
        mel_aug[f0:f0+f, :] = mel_aug.min()
    return mel_aug

# -------------------------
# 4) Label map helpers
# -------------------------
def build_label_map(labels: List[str]) -> Tuple[dict, dict]:
    uniq = sorted(set(labels))
    label2idx = {lab: i for i, lab in enumerate(uniq)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label

# -------------------------
# 5) PyTorch Dataset
# -------------------------
class BirdDataset(Dataset):
    def __init__(self,
                 index_csv: str,
                 label2idx: Optional[dict] = None,
                 sr: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 min_duration: float = 0.2,
                 max_duration: Optional[float] = None,
                 augment: bool = False,
                 normalize: bool = True):
        """
        index_csv: path to CSV with columns ['path','label']
        If label2idx None, will build from CSV
        """
        self.df = pd.read_csv(index_csv)
        if label2idx is None:
            self.label2idx, self.idx2label = build_label_map(self.df['label'].tolist())
        else:
            self.label2idx = label2idx
            # build inverse map if possible
            self.idx2label = {v:k for k,v in label2idx.items()}
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        label_name = row['label']
        label = self.label2idx[label_name]

        # load whole file (you can also sample random crops for augmentation)
        y, sr = load_audio(path, sr=self.sr)
        # trim silence (optional)
        y = trim_silence(y, top_db=40)
        # ensure minimum duration: pad if too short
        min_samples = int(self.min_duration * self.sr)
        if len(y) < min_samples:
            pad_len = min_samples - len(y)
            y = np.pad(y, (0, pad_len), mode='constant')

        # optional augment chain
        if self.augment:
            # random apply add noise
            if random.random() < 0.5:
                y = add_white_noise(y, snr_db=random.uniform(10, 30))
            # random time shift up to 0.1s
            if random.random() < 0.5:
                y = time_shift(y, shift_max=0.05, sr=self.sr)

        # highpass filter to remove low-frequency noise
        y = highpass_filter(y, sr=self.sr, cutoff=300.0)

        # normalization
        if self.normalize:
            y = normalize_audio(y)

        # mel spectrogram
        mel_db = compute_mel_spectrogram(y, sr=self.sr,
                                         n_mels=self.n_mels,
                                         n_fft=self.n_fft,
                                         hop_length=self.hop_length)

        # optionally spec augment on mel (prob)
        if self.augment and random.random() < 0.5:
            mel_db = spec_augment(mel_db, time_mask_param=30, freq_mask_param=12,
                                  num_time_masks=1, num_freq_masks=1)

        # convert to torch tensor, add channel dim
        mel_tensor = torch.from_numpy(mel_db).unsqueeze(0)  # (1, n_mels, T)
        sample = {
            'mel': mel_tensor,          # float tensor
            'label': torch.tensor(label, dtype=torch.long),
            'path': path,
            'orig_len': torch.tensor(len(y), dtype=torch.long)
        }
        return sample

# -------------------------
# 6) collate_fn: pad along time axis
# -------------------------
def pad_collate_fn(batch):
    """
    batch: list of samples from BirdDataset
    Pads mel time dimension to the max T in batch.
    Returns:
      mels: (B, 1, n_mels, T_max)
      labels: (B,)
      masks: (B, T_max)  # 1 for valid frames, 0 for padded
      paths: list[str]
    """
    mels = [s['mel'] for s in batch]
    labels = torch.stack([s['label'] for s in batch])
    paths = [s['path'] for s in batch]
    orig_lens = [s['mel'].shape[-1] for s in batch]
    B = len(mels)
    n_mels = mels[0].shape[1]
    T_max = max([m.shape[-1] for m in mels])
    mels_padded = torch.zeros((B, 1, n_mels, T_max), dtype=mels[0].dtype)
    masks = torch.zeros((B, T_max), dtype=torch.bool)
    for i, m in enumerate(mels):
        T = m.shape[-1]
        mels_padded[i, :, :, :T] = m
        masks[i, :T] = 1
    return {
        'mels': mels_padded,
        'labels': labels,
        'masks': masks,
        'paths': paths
    }

# -------------------------
# 7) DataLoader 使用示例
# -------------------------
def example_usage(index_csv: str, batch_size: int = 16, num_workers: int = 4):
    # build dataset
    dataset = BirdDataset(index_csv=index_csv, augment=True)
    # build dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=pad_collate_fn, pin_memory=True)
    # iterate one epoch
    for i, batch in enumerate(loader):
        mels = batch['mels']      # (B,1,n_mels,T)
        labels = batch['labels']  # (B,)
        masks = batch['masks']    # (B,T)
        # now you can feed to your model. Example:
        print("batch", i, "mels", mels.shape, "labels", labels.shape, "masks", masks.shape)
        if i >= 2:
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=r'D:\鸟种库', help='root data folder')
    parser.add_argument('--index', type=str, default='index.csv', help='output index CSV or input CSV')
    parser.add_argument('--build_index', action='store_true', help='scan root and build index CSV')
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    if args.build_index:
        build_index(args.root, args.index)

    if args.process:
        index_file = "index.csv"
        save_root = r"D:\鸟类库_processed"
        os.makedirs(save_root, exist_ok=True)

        df = pd.read_csv(index_file)

         # 按类别处理
        for label in df['label'].unique():
            df_label = df[df['label'] == label]
            mels_list = []

            for path in df_label['path']:
                y, sr = load_audio(path, max_len_sec=1.0)
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=128,
                    power=2
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mels_list.append(mel_db)

        # 合并成一个 numpy array
        mels_array = np.stack(mels_list, axis=0)  # [num_samples, 128, time_frames]
        save_path = os.path.join(save_root, f"{label}.npz")
        np.savez_compressed(save_path, mels=mels_array)
        print(f"{label} 保存完成, shape={mels_array.shape}")

    print("Creating dataloader from:", args.index)
    example_usage(index_csv=args.index, batch_size=args.batch)
