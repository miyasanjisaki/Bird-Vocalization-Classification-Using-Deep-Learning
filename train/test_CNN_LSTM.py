import os
from collections import Counter
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .CNN_LSTM import CNN_LSTM
from bird_preprocess.init import (
    trim_silence,
    highpass_filter,
    normalize_audio,
    compute_mel_spectrogram,
)


# -------------------------------
# 参数配置
# -------------------------------
sr = 16000
segment_sec = 1.0   # 每段分析长度（秒）
hop_sec = 0.5        # 滑动步长（秒）
DEFAULT_MODEL_PATH = os.environ.get(
    "BIRD_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "cnn_lstm_npz.pth"),
)
DEFAULT_DATA_ROOT = os.environ.get("BIRD_AUDIO_ROOT", r"D:\鸟类测试音频")
DEFAULT_PROCESSED_ROOT = os.environ.get("BIRD_PROCESSED_ROOT", r"D:\鸟类库_processed")

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# 加载标签映射
# -------------------------------

def load_label_maps(processed_root: str) -> Tuple[List[str], Dict[int, str]]:
    label_files = sorted(
        [f for f in os.listdir(processed_root) if f.endswith(".npz")]
    )
    if not label_files:
        raise FileNotFoundError(
            f"在 {processed_root} 中未找到 .npz 标签文件，请确认预处理数据路径是否正确。"
        )
    label_names = [os.path.splitext(f)[0] for f in label_files]
    idx2label = {i: lab for i, lab in enumerate(label_names)}
    return label_names, idx2label

# -------------------------------
# 加载模型
# -------------------------------

def load_model(model_path: str, num_classes: int) -> CNN_LSTM:
    model = CNN_LSTM(n_mels=128, num_classes=num_classes, hidden_dim=256)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# -------------------------------
# 分段预测函数
# -------------------------------
def predict_audio(
    audio_path: str,
    model: CNN_LSTM,
    idx2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """对单个音频文件进行滑窗预测并返回包含时间戳的事件列表。"""

    y, _sr = librosa.load(audio_path, sr=sr, mono=True)
    y = trim_silence(y)
    y = highpass_filter(y, sr, cutoff=300)
    y = normalize_audio(y)

    seg_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)

    # 若信号过短，补零至一个窗口长度
    if len(y) < seg_len:
        pad_len = seg_len - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")

    events: List[Dict[str, Any]] = []
    for start in range(0, max(len(y) - seg_len + 1, 1), hop_len):
        end = min(start + seg_len, len(y))
        seg = y[start:end]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)), mode="constant")

        mel = compute_mel_spectrogram(seg, sr=sr)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(mel_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx].item())

        events.append({
            "start_sec": start / sr,
            "end_sec": end / sr,
            "label_idx": pred_idx,
            "label_name": idx2label[pred_idx],
            "confidence": confidence,
        })

    return events

# -------------------------------
# 主函数：遍历文件夹预测
# -------------------------------
def main(
    input_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    processed_root: str = DEFAULT_PROCESSED_ROOT,
):
    label_names, idx2label = load_label_maps(processed_root)
    model = load_model(model_path, num_classes=len(label_names))
    print(f"模型已加载，共 {len(label_names)} 类鸟叫。")

    audio_paths = librosa.util.find_files(input_dir, ext=['wav', 'mp3', 'flac'])
    print(f"检测到 {len(audio_paths)} 个音频文件")

    counter = Counter()
    total_calls = 0

    events_records = []

    for path in audio_paths:
        events = predict_audio(path, model=model, idx2label=idx2label)
        for event in events:
            counter[event["label_name"]] += 1
        total_calls += len(events)

        for event in events:
            label = event.get("label", event.get("label_name"))  # 兼容两种键
            events_records.append({
                "audio_file": os.path.basename(path),
                "start_sec": event["start_sec"],
                "end_sec": event["end_sec"],
                "label": label,  # ← 统一成 'label'
                "confidence": event["confidence"],
            })

    print(f"\n一共检测到 {total_calls} 声鸟叫：\n")
    for bird, cnt in counter.items():
        print(f"{bird}： {cnt} 声")

    # 可选：保存结果
    import pandas as pd

    df = pd.DataFrame(list(counter.items()), columns=["鸟种", "叫声次数"])
    df.loc[len(df)] = ["总计", total_calls]
    df.to_csv("bird_count.csv", index=False, encoding="utf-8-sig")
    print("\n已将统计结果保存为 bird_count.csv")

    if events_records:
        events_df = pd.DataFrame(events_records)
        events_df = events_df.sort_values(["audio_file", "start_sec"]).reset_index(drop=True)
        events_df.to_csv("bird_events.csv", index=False, encoding="utf-8-sig")
        print("事件明细已保存为 bird_events.csv")
    else:
        print("未检测到任何事件，未生成 bird_events.csv")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="批量分析音频文件并输出鸟类叫声统计与时间线。"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_DATA_ROOT,
        help="要分析的音频文件夹路径（默认取 BIRD_AUDIO_ROOT 或脚本内默认值）",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="模型权重 .pth 文件路径（默认取 BIRD_MODEL_PATH 或脚本内默认值）",
    )
    parser.add_argument(
        "--processed-root",
        default=DEFAULT_PROCESSED_ROOT,
        help="包含标签 .npz 文件的目录（默认取 BIRD_PROCESSED_ROOT 或脚本内默认值）",
    )

    args = parser.parse_args()
    main(args.input_dir, model_path=args.model_path, processed_root=args.processed_root)
