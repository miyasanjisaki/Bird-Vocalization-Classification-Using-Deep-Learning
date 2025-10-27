import os
import torch
import numpy as np
import librosa
from collections import Counter

from CNN_LSTM import CNN_LSTM
from init import load_audio, trim_silence, highpass_filter, normalize_audio, compute_mel_spectrogram


# -------------------------------
# 参数配置
# -------------------------------
sr = 16000
segment_sec = 1.0   # 每段分析长度（秒）
hop_sec = 0.5        # 滑动步长（秒）
model_path = "cnn_lstm_npz.pth"
data_root = r"D:\鸟类测试音频"
processed_root = r"D:\鸟类库_processed"   # 用来加载 label 名称
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# 加载标签映射
# -------------------------------
label_files = sorted([f for f in os.listdir(processed_root) if f.endswith(".npz")])
label_names = [os.path.splitext(f)[0] for f in label_files]
label2idx = {lab: i for i, lab in enumerate(label_names)}
idx2label = {i: lab for lab, i in label2idx.items()}

# -------------------------------
# 加载模型
# -------------------------------
num_classes = len(label2idx)
model = CNN_LSTM(n_mels=128, num_classes=num_classes, hidden_dim=256)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"模型已加载，共 {num_classes} 类鸟叫。")

# -------------------------------
# 分段预测函数
# -------------------------------
def predict_audio(audio_path):
    y, sr = load_audio(audio_path, sr=sr, max_len_sec=None)
    y = trim_silence(y)
    y = highpass_filter(y, sr, cutoff=300)
    y = normalize_audio(y)

    seg_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)
    preds = []

    for start in range(0, len(y) - seg_len + 1, hop_len):
        seg = y[start:start + seg_len]
        mel = compute_mel_spectrogram(seg, sr=sr)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(mel_tensor)
            pred = logits.argmax(1).item()
            preds.append(pred)

    return preds

# -------------------------------
# 主函数：遍历文件夹预测
# -------------------------------
def main(input_dir):
    audio_paths = librosa.util.find_files(input_dir, ext=['wav', 'mp3', 'flac'])
    print(f"检测到 {len(audio_paths)} 个音频文件")

    counter = Counter()
    total_calls = 0

    for path in audio_paths:
        preds = predict_audio(path)
        for p in preds:
            counter[idx2label[p]] += 1
        total_calls += len(preds)

    print(f"\n一共检测到 {total_calls} 声鸟叫：\n")
    for bird, cnt in counter.items():
        print(f"{bird}： {cnt} 声")

    # 可选：保存结果
    import pandas as pd
    df = pd.DataFrame(list(counter.items()), columns=["鸟种", "叫声次数"])
    df.loc[len(df)] = ["总计", total_calls]
    df.to_csv("bird_count.csv", index=False, encoding="utf-8-sig")
    print("\n已将统计结果保存为 bird_count.csv")

if __name__ == "__main__":
    main(data_root)
