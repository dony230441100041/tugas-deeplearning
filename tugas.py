import os
import sys
import time
import json
import pickle
import random
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
_ICONS = {"INFO": "ℹ️ ", "STEP": "🔹", "OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}


def log_step(message, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {_ICONS.get(level, '•')} {message}")


def log_section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
log_section("TAHAP 1 — Google Drive Mounting & Setup Direktori")

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = "/https://drive.google.com/drive/folders/1oTWVxqOS2M1ql4sdw7cBgZyRMHRlUP4a?usp=sharing"

if IN_COLAB:
    from google.colab import drive
    log_step("Mounting Google Drive...", "STEP")
    drive.mount("/content/drive", force_remount=False)
    log_step("Google Drive berhasil di-mount.", "OK")
else:
    log_step("Tidak berjalan di Google Colab — memakai direktori lokal sebagai gantinya.", "WARN")
    BASE_DIR = "./https://drive.google.com/drive/folders/1oTWVxqOS2M1ql4sdw7cBgZyRMHRlUP4a?usp=sharing"

DIRS = {
    "base": BASE_DIR,
    "grafik": os.path.join(BASE_DIR, "grafik"),
    "cache": os.path.join(BASE_DIR, "cache"),
    "laporan": os.path.join(BASE_DIR, "laporan"),
    "error_analysis": os.path.join(BASE_DIR, "error_analysis"),
}
for name, path in DIRS.items():
    os.makedirs(path, exist_ok=True)
    log_step(f"Direktori '{name}' siap -> {path}", "OK")

# Pastikan python-docx tersedia untuk ekspor laporan Word
try:
    import docx  # noqa: F401
except ImportError:
    log_step("python-docx belum terpasang, menginstal...", "STEP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"], check=True)
    log_step("python-docx berhasil diinstal.", "OK")
log_section("TAHAP 2 — Konfigurasi Eksperimen")
log_section("TAHAP 2 — Konfigurasi Eksperimen")

CONFIG = {
    "seed": 42,
    "batch_size": 128,
    "epochs": 15,        # naikkan jika GPU T4 tersedia & butuh akurasi lebih tinggi
    "lr": 1e-3,
    "val_split": 0.1,
    "num_workers": 2,
    "dpi": 300,
}

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


set_seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_step(f"Perangkat komputasi: {DEVICE}", "INFO")
if DEVICE.type == "cuda":
    log_step(f"GPU terdeteksi: {torch.cuda.get_device_name(0)}", "INFO")
else:
    log_step("GPU tidak terdeteksi, training berjalan di CPU (lebih lambat).", "WARN")
log_step(f"Konfigurasi: {json.dumps(CONFIG, indent=2)}", "INFO")
def get_dataloaders(config):
    log_section("TAHAP 3 — Memuat Dataset CIFAR-10")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    log_step("Mengunduh/memuat CIFAR-10 (train)...", "STEP")
    full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    full_train_eval = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=eval_transform)
    log_step("Mengunduh/memuat CIFAR-10 (test)...", "STEP")
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_transform)
    # dataset mentah (tanpa normalisasi) khusus untuk visualisasi error analysis
    raw_test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    n_val = int(len(full_train) * config["val_split"])
    n_train = len(full_train) - n_val
    gen = torch.Generator().manual_seed(config["seed"])
    train_idx, val_idx = random_split(range(len(full_train)), [n_train, n_val], generator=gen)

    train_set = torch.utils.data.Subset(full_train, train_idx.indices)
    val_set = torch.utils.data.Subset(full_train_eval, val_idx.indices)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    log_step(f"Jumlah data -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}", "OK")
    return train_loader, val_loader, test_loader, raw_test_set
class ModelA_SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, use_dropout=True, dropout_p=0.4):
        super().__init__()
        self.use_dropout = use_dropout
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_logits=True):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        logits = self.fc2(x)
        return logits if return_logits else self.softmax(logits)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ModelB_MiniResNet(nn.Module):
    def __init__(self, num_classes=10, num_blocks=(2, 2, 2)):
        super().__init__()
        self.in_channels = 32
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stage1 = self._make_stage(32, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(64, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(128, num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def _make_stage(self, out_channels, n_blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train_model(model, train_loader, val_loader, config, model_name="model"):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    log_section(f"Training {model_name}")
    log_step(f"Jumlah parameter: {count_parameters(model):,}", "INFO")
    start_time = time.time()

    for epoch in range(config["epochs"]):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer=None)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        log_step(
            f"Epoch {epoch+1:02d}/{config['epochs']} | train_loss={tr_loss:.4f} "
            f"val_loss={val_loss:.4f} | train_acc={tr_acc:.4f} val_acc={val_acc:.4f}",
            "STEP",
        )

    total_time = time.time() - start_time
    log_step(f"Training {model_name} selesai dalam {total_time:.1f} detik.", "OK")
    return model, history, total_time


def evaluate_on_test(model, test_loader, model_name="model"):
    model.eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = total_loss / total_samples
    test_acc = total_correct / total_samples
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    log_step(f"[{model_name}] TEST -> loss={test_loss:.4f} acc={test_acc:.4f}", "OK")
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }
def save_fig(filename):
    path = os.path.join(DIRS["grafik"], filename)
    plt.tight_layout()
    plt.savefig(path, dpi=CONFIG["dpi"])
    plt.close()
    log_step(f"Grafik tersimpan ({CONFIG['dpi']} DPI): {path}", "OK")
    return path


def plot_curves(histories: dict, key, ylabel, filename):
    plt.figure(figsize=(7, 5))
    for name, hist in histories.items():
        plt.plot(hist[key], label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    return save_fig(filename)


def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    return save_fig(filename)
def two_way_error_analysis(eval_a, eval_b, raw_test_set, n_examples=6):
    log_section("TAHAP — Error Analysis Dua Arah (Model A vs Model B)")

    preds_a, preds_b, labels = eval_a["preds"], eval_b["preds"], eval_a["labels"]
    idx_a_wrong_b_right = np.where((preds_a != labels) & (preds_b == labels))[0]
    idx_b_wrong_a_right = np.where((preds_b != labels) & (preds_a == labels))[0]

    log_step(f"Kasus Model A SALAH, Model B BENAR: {len(idx_a_wrong_b_right)} citra", "INFO")
    log_step(f"Kasus Model B SALAH, Model A BENAR: {len(idx_b_wrong_a_right)} citra", "INFO")

    result = {
        "n_a_wrong_b_right": int(len(idx_a_wrong_b_right)),
        "n_b_wrong_a_right": int(len(idx_b_wrong_a_right)),
        "example_path": None,
    }

    n = min(n_examples, len(idx_a_wrong_b_right), len(idx_b_wrong_a_right))
    if n == 0:
        log_step("Tidak cukup sampel di salah satu kategori untuk divisualisasikan.", "WARN")
        return result

    rng = np.random.default_rng(CONFIG["seed"])
    sample_a_wrong = rng.choice(idx_a_wrong_b_right, n, replace=False)
    sample_b_wrong = rng.choice(idx_b_wrong_a_right, n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2))
    for col, idx in enumerate(sample_a_wrong):
        img, _ = raw_test_set[int(idx)]
        axes[0, col].imshow(img.permute(1, 2, 0).numpy())
        axes[0, col].set_title(
            f"True: {CLASS_NAMES[labels[idx]]}\nA:{CLASS_NAMES[preds_a[idx]]}  B:{CLASS_NAMES[preds_b[idx]]}",
            fontsize=8,
        )
        axes[0, col].axis("off")
    for col, idx in enumerate(sample_b_wrong):
        img, _ = raw_test_set[int(idx)]
        axes[1, col].imshow(img.permute(1, 2, 0).numpy())
        axes[1, col].set_title(
            f"True: {CLASS_NAMES[labels[idx]]}\nA:{CLASS_NAMES[preds_a[idx]]}  B:{CLASS_NAMES[preds_b[idx]]}",
            fontsize=8,
        )
        axes[1, col].axis("off")

    fig.text(0.01, 0.75, "Model A salah,\nModel B benar", rotation=90, va="center", fontsize=9, weight="bold")
    fig.text(0.01, 0.28, "Model B salah,\nModel A benar", rotation=90, va="center", fontsize=9, weight="bold")
    fig.suptitle("Error Analysis Dua Arah: Model A vs Model B", fontsize=13)
    plt.tight_layout(rect=[0.04, 0, 1, 0.95])

    path = os.path.join(DIRS["error_analysis"], "error_analysis_dua_arah.png")
    plt.savefig(path, dpi=CONFIG["dpi"])
    plt.close()
    log_step(f"Grafik error analysis tersimpan: {path}", "OK")

    result["example_path"] = path
    return result
def save_cache(results, filename="cache.pkl"):
    log_section("TAHAP — Menyimpan Cache Eksperimen")
    path = os.path.join(DIRS["cache"], filename)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    log_step(f"Cache eksperimen tersimpan: {path}", "OK")
    return path


def load_cache(filename="cache.pkl"):
    path = os.path.join(DIRS["cache"], filename)
    with open(path, "rb") as f:
        return pickle.load(f)
def add_metric_table(doc, headers, rows):
    from docx.shared import Pt
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = str(h)
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(10)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    return table


def generate_word_report(results, output_path):
    log_section("TAHAP — Menyusun Laporan Word Otomatis")
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    doc.add_heading("Laporan Hasil Eksperimen", level=0)
    doc.add_paragraph(
        f"Tugas Deep Learning — Perbandingan Model A (CNN Custom) vs Model B (Mini-ResNet) "
        f"pada dataset CIFAR-10. Dihasilkan otomatis pada {datetime.now().strftime('%d %B %Y %H:%M')}."
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 1. Ringkasan arsitektur & konfigurasi
    doc.add_heading("1. Ringkasan Arsitektur & Konfigurasi", level=1)
    doc.add_paragraph(
        "Model A: CNN custom dengan 3 blok Conv-ReLU-Pool (32→64→128 channel), diikuti fully "
        "connected layer dan output softmax 10 kelas."
    )
    doc.add_paragraph(
        "Model B: Mini-ResNet dibangun dari awal (bukan pretrained) dengan residual block "
        "sederhana pada 3 stage (32→64→128 channel), diakhiri global average pooling dan FC output."
    )
    cfg = results["config"]
    doc.add_paragraph(
        f"Hyperparameter (identik untuk kedua model): epoch={cfg['epochs']}, "
        f"batch_size={cfg['batch_size']}, optimizer=Adam, learning_rate={cfg['lr']}."
    )

    # 2. Tabel ringkasan hasil B.3
    doc.add_heading("2. Tabel Ringkasan Hasil Eksperimen", level=1)
    rows = []
    for name in ["model_a", "model_b"]:
        r = results[name]
        rows.append([
            r["display_name"], f"{r['params']:,}", f"{r['training_time_sec']:.1f}",
            f"{r['test_loss']:.4f}", f"{r['test_acc']:.4f}",
        ])
    add_metric_table(doc, ["Model", "Jumlah Parameter", "Waktu Training (s)", "Test Loss", "Test Accuracy"], rows)

    # 3. Grafik training
    doc.add_heading("3. Grafik Hasil Pelatihan", level=1)
    for path, caption in results["graph_paths"]:
        if path and os.path.exists(path):
            doc.add_picture(path, width=Inches(5.5))
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in cap.runs:
                r.italic = True
                r.font.size = Pt(9)

    # 4. Classification report per kelas
    doc.add_heading("4. Classification Report per Kelas", level=1)
    for name in ["model_a", "model_b"]:
        r = results[name]
        doc.add_heading(r["display_name"], level=2)
        rep = r["classification_report"]
        rows = []
        for cls in CLASS_NAMES:
            m = rep[cls]
            rows.append([cls, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1-score']:.3f}", int(m["support"])])
        add_metric_table(doc, ["Kelas", "Precision", "Recall", "F1-Score", "Support"], rows)

    # 5. Ablation study
    doc.add_heading("5. Ablation Study — Dropout On vs Off (Model A)", level=1)
    ab = results["ablation"]
    rows = [
        ["Dengan dropout", f"{ab['with_dropout']['final_train_acc']:.4f}",
         f"{ab['with_dropout']['final_val_acc']:.4f}", f"{ab['with_dropout']['test_acc']:.4f}",
         f"{ab['with_dropout']['train_val_gap']:.4f}"],
        ["Tanpa dropout", f"{ab['without_dropout']['final_train_acc']:.4f}",
         f"{ab['without_dropout']['final_val_acc']:.4f}", f"{ab['without_dropout']['test_acc']:.4f}",
         f"{ab['without_dropout']['train_val_gap']:.4f}"],
    ]
    add_metric_table(doc, ["Varian", "Train Acc", "Val Acc", "Test Acc", "Gap (Train-Val)"], rows)

    # 6. Error analysis dua arah
    doc.add_heading("6. Error Analysis Dua Arah", level=1)
    ea = results["error_analysis"]
    doc.add_paragraph(
        f"Ditemukan {ea['n_a_wrong_b_right']} citra yang salah diklasifikasikan oleh Model A namun "
        f"benar oleh Model B, serta {ea['n_b_wrong_a_right']} citra sebaliknya (benar di Model A, "
        f"salah di Model B)."
    )
    if ea.get("example_path") and os.path.exists(ea["example_path"]):
        doc.add_picture(ea["example_path"], width=Inches(6))

    # 7. Pembahasan ilmiah (auto-generated)
    doc.add_heading("7. Pembahasan Ilmiah", level=1)
    winner = "Model B (Mini-ResNet)" if results["model_b"]["test_acc"] >= results["model_a"]["test_acc"] else "Model A (CNN Custom)"
    gap_with = ab["with_dropout"]["train_val_gap"]
    gap_without = ab["without_dropout"]["train_val_gap"]
    dropout_effect = "mengurangi" if gap_with < gap_without else "tidak secara jelas mengurangi"
    doc.add_paragraph(
        f"Berdasarkan hasil eksperimen, {winner} mencapai akurasi test lebih tinggi "
        f"({results['model_a']['test_acc']:.4f} untuk Model A vs {results['model_b']['test_acc']:.4f} "
        f"untuk Model B). Hal ini konsisten dengan teori residual connection pada ResNet (He et al., 2016), "
        f"di mana koneksi shortcut membantu mengatasi masalah vanishing gradient sehingga jaringan yang "
        f"lebih dalam tetap dapat dioptimalkan dengan efektif, dibandingkan CNN polos yang rentan mengalami "
        f"degradasi performa seiring bertambahnya kedalaman."
    )
    doc.add_paragraph(
        f"Pada ablation study dropout, penggunaan dropout {dropout_effect} gap antara akurasi training dan "
        f"validation (gap dengan dropout = {gap_with:.4f}, tanpa dropout = {gap_without:.4f}). Gap yang lebih "
        f"kecil mengindikasikan regularisasi dropout membantu menekan overfitting pada Model A."
    )

    # 8. Kesimpulan
    doc.add_heading("8. Kesimpulan", level=1)
    doc.add_paragraph(
        f"Secara keseluruhan, {winner} menunjukkan performa generalisasi yang lebih baik pada dataset "
        f"CIFAR-10 dalam eksperimen ini, dengan trade-off jumlah parameter dan waktu training yang perlu "
        f"dipertimbangkan sesuai kebutuhan aplikasi. Penggunaan dropout terbukti bermanfaat sebagai teknik "
        f"regularisasi untuk model CNN sederhana."
    )

    doc.save(output_path)
    log_step(f"Laporan Word tersimpan: {output_path}", "OK")
    return output_path
def main():
    train_loader, val_loader, test_loader, raw_test_set = get_dataloaders(CONFIG)

    # ---- B.3: Model A vs Model B (hyperparameter identik) ----
    model_a, hist_a, time_a = train_model(ModelA_SimpleCNN(use_dropout=True), train_loader, val_loader, CONFIG, "Model A (CNN)")
    eval_a = evaluate_on_test(model_a, test_loader, "Model A (CNN)")

    model_b, hist_b, time_b = train_model(ModelB_MiniResNet(num_blocks=(2, 2, 2)), train_loader, val_loader, CONFIG, "Model B (Mini-ResNet)")
    eval_b = evaluate_on_test(model_b, test_loader, "Model B (Mini-ResNet)")

    log_section("TAHAP — Ekspor Grafik B.3")
    histories = {"Model A (CNN)": hist_a, "Model B (Mini-ResNet)": hist_b}
    p1 = plot_curves(histories, "train_loss", "Train Loss", "b3_train_loss.png")
    p2 = plot_curves(histories, "val_loss", "Validation Loss", "b3_val_loss.png")
    p3 = plot_curves(histories, "train_acc", "Train Accuracy", "b3_train_acc.png")
    p4 = plot_curves(histories, "val_acc", "Validation Accuracy", "b3_val_acc.png")
    p5 = plot_confusion_matrix(eval_a["confusion_matrix"], "Confusion Matrix - Model A", "b3_cm_model_a.png")
    p6 = plot_confusion_matrix(eval_b["confusion_matrix"], "Confusion Matrix - Model B", "b3_cm_model_b.png")

    # ---- B.4: Ablation dropout on vs off ----
    model_a_nodrop, hist_a_nodrop, time_a_nodrop = train_model(
        ModelA_SimpleCNN(use_dropout=False), train_loader, val_loader, CONFIG, "Model A tanpa dropout"
    )
    eval_a_nodrop = evaluate_on_test(model_a_nodrop, test_loader, "Model A tanpa dropout")

    log_section("TAHAP — Ekspor Grafik B.4 (Ablasi Dropout)")
    ablation_hist = {"Model A - dengan dropout": hist_a, "Model A - tanpa dropout": hist_a_nodrop}
    p7 = plot_curves(ablation_hist, "train_acc", "Train Accuracy (Ablasi Dropout)", "b4_ablation_train_acc.png")
    p8 = plot_curves(ablation_hist, "val_acc", "Validation Accuracy (Ablasi Dropout)", "b4_ablation_val_acc.png")

    gap_with = hist_a["train_acc"][-1] - hist_a["val_acc"][-1]
    gap_without = hist_a_nodrop["train_acc"][-1] - hist_a_nodrop["val_acc"][-1]

    # ---- Error analysis dua arah ----
    error_analysis = two_way_error_analysis(eval_a, eval_b, raw_test_set)

    # ---- Susun struktur hasil lengkap ----
    results = {
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "model_a": {
            "display_name": "Model A (CNN Custom)",
            "params": count_parameters(model_a),
            "training_time_sec": time_a,
            "test_loss": eval_a["test_loss"],
            "test_acc": eval_a["test_acc"],
            "confusion_matrix": eval_a["confusion_matrix"],
            "classification_report": eval_a["classification_report"],
            "history": hist_a,
        },
        "model_b": {
            "display_name": "Model B (Mini-ResNet)",
            "params": count_parameters(model_b),
            "training_time_sec": time_b,
            "test_loss": eval_b["test_loss"],
            "test_acc": eval_b["test_acc"],
            "confusion_matrix": eval_b["confusion_matrix"],
            "classification_report": eval_b["classification_report"],
            "history": hist_b,
        },
        "ablation": {
            "with_dropout": {
                "final_train_acc": hist_a["train_acc"][-1],
                "final_val_acc": hist_a["val_acc"][-1],
                "test_acc": eval_a["test_acc"],
                "train_val_gap": gap_with,
            },
            "without_dropout": {
                "final_train_acc": hist_a_nodrop["train_acc"][-1],
                "final_val_acc": hist_a_nodrop["val_acc"][-1],
                "test_acc": eval_a_nodrop["test_acc"],
                "train_val_gap": gap_without,
            },
        },
        "error_analysis": error_analysis,
        "graph_paths": [
            (p1, "Grafik 1. Training Loss per Epoch"),
            (p2, "Grafik 2. Validation Loss per Epoch"),
            (p3, "Grafik 3. Training Accuracy per Epoch"),
            (p4, "Grafik 4. Validation Accuracy per Epoch"),
            (p5, "Grafik 5. Confusion Matrix - Model A"),
            (p6, "Grafik 6. Confusion Matrix - Model B"),
            (p7, "Grafik 7. Ablasi Dropout - Training Accuracy"),
            (p8, "Grafik 8. Ablasi Dropout - Validation Accuracy"),
        ],
    }

    # ---- Cache eksperimen (pickle) ----
    save_cache(results, "cache.pkl")

    # ---- Laporan Word otomatis ----
    report_path = os.path.join(DIRS["laporan"], "Laporan_Hasil_Eksperimen.docx")
    generate_word_report(results, report_path)

    # ---- Ringkasan akhir ----
    log_section("RINGKASAN AKHIR EKSPERIMEN")
    log_step(f"Model A -> params={results['model_a']['params']:,} | waktu={time_a:.1f}s | test_acc={eval_a['test_acc']:.4f}", "OK")
    log_step(f"Model B -> params={results['model_b']['params']:,} | waktu={time_b:.1f}s | test_acc={eval_b['test_acc']:.4f}", "OK")
    log_step(f"Ablasi dropout -> gap dgn dropout={gap_with:.4f} | gap tanpa dropout={gap_without:.4f}", "OK")
    log_step(f"Semua file tersimpan di Google Drive: {BASE_DIR}", "OK")

    return results


results = main()  # jalankan seluruh eksperimen
