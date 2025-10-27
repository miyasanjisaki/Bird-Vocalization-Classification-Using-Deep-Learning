import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# ============================
# 1) Dataset：读取 npz 文件
# ============================
class BirdNPZDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        self.label_names = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
        self.label2idx = {lab: i for i, lab in enumerate(self.label_names)}

        self.data = []
        for f, label in zip(self.files, self.label_names):
            arr = np.load(f)["mels"]  # (N, 128, T)
            for mel in arr:
                self.data.append((mel, self.label2idx[label]))
        print(f"已加载 {len(self.data)} 条样本，共 {len(self.label2idx)} 类。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel, label = self.data[idx]
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1,128,T)
        return {"mel": mel_tensor, "label": torch.tensor(label, dtype=torch.long)}

def pad_collate_fn(batch):
    mels = [b["mel"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    T_max = max([m.shape[-1] for m in mels])
    B = len(batch)
    n_mels = mels[0].shape[1]
    mels_padded = torch.zeros((B, 1, n_mels, T_max))
    masks = torch.zeros((B, T_max))
    for i, m in enumerate(mels):
        T = m.shape[-1]
        mels_padded[i, :, :, :T] = m
        masks[i, :T] = 1
    return {"mels": mels_padded, "labels": labels, "masks": masks}


class CNN_LSTM(nn.Module):
    def __init__(self, n_mels=128, num_classes=20, hidden_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   # n_mels -> n_mels/2

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2))    # n_mels -> n_mels/4
        )

        # ↓ 用一个 dummy tensor 自动计算 CNN 输出特征维度
        dummy = torch.zeros(1, 1, n_mels, 19)
        with torch.no_grad():
            dummy_out = self.cnn(dummy)     # (1, C, F, T)
            _, C, F, T = dummy_out.shape
            lstm_input_dim = C * F          # 自动算出实际输入维度
            print(f"[Init] CNN 输出维度: {C}x{F}x{T}, LSTM输入维度={lstm_input_dim}")

        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, mels, masks=None):
        x = self.cnn(mels)  # (B,C,F,T)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B,T,C,F)
        x = x.reshape(B, T, C * F)  # (B, T, features=2048)

        out, _ = self.lstm(x)  # LSTM input_size=2048
        out_mean = out.mean(1)
        logits = self.fc(out_mean)
        return logits


# ============================
# 3) 训练与验证函数
# ============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0
    for batch in loader:
        mels = batch["mels"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(mels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        acc = (preds == labels).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss / len(loader), total_acc / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        mels = batch["mels"].to(device)
        labels = batch["labels"].to(device)

        logits = model(mels)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    return acc, report


# ============================
# 4) 主流程
# ============================
def main():
    root_dir = r"D:\鸟类库_processed"
    batch_size = 64
    num_epochs = 25
    lr = 5e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BirdNPZDataset(root_dir)
    n = len(dataset)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    split = int(n * 0.8)
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    num_classes = len(dataset.label2idx)
    model = CNN_LSTM(n_mels=128, num_classes=num_classes, hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # ----------------
    # 训练循环
    # ----------------
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, report = evaluate(model, val_loader, device)
        print(f"\n[Epoch {epoch+1}/{num_epochs}] TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f}")
        print(report)

    torch.save(model.state_dict(), "cnn_lstm_npz.pth")
    print("模型已保存至 cnn_lstm_npz.pth")

if __name__ == "__main__":
    main()
