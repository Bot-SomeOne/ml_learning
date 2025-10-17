
import os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ===== Config =====
DATA_DIR = Path("./data")
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"

BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
IMG_SIZE = 224
MODEL_OUT = "catdog_resnet18_best.pt"
SEED = 42

def seed_everything(seed=42):
    import torch
    import numpy as np, random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(SEED)

    # --- Device ---
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # --- Data transforms ---
    from torchvision import transforms
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(0.15, 0.15, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    valid_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tfms)
    valid_ds = datasets.ImageFolder(str(VALID_DIR), transform=valid_tfms)
    assert train_ds.classes == valid_ds.classes
    class_names = train_ds.classes
    num_classes = len(class_names)

    # --- DataLoader knobs for macOS/MPS ---
    is_cuda = (device.type == "cuda")
    is_mps  = (device.type == "mps")
    NUM_WORKERS = 0 if not is_cuda else (os.cpu_count() or 4)
    PIN_MEMORY  = True if is_cuda else False
    PERSISTENT  = True if (is_cuda and NUM_WORKERS > 0) else False

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT,
    )

    # --- Model ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # AMP chỉ dành cho CUDA; MPS/CPU tắt hẳn
    if is_cuda:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler(device_type="cuda")
        autocast_ctx = lambda: autocast(device_type="cuda")
    else:
        scaler = None
        # no-op context manager
        from contextlib import contextmanager
        @contextmanager
        def _noop(): yield
        autocast_ctx = _noop

    def run_epoch(dataloader, train=True):
        model.train(mode=train)
        epoch_loss, correct, total = 0.0, 0, 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast_ctx():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return epoch_loss / total, correct / total

    best_acc, best_state, no_improve = 0.0, None, 0
    history = {"train_acc": [], "valid_acc": [], "train_loss": [], "valid_loss": []}

    t0 = time.time()
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(train_loader, True)
        va_loss, va_acc = run_epoch(valid_loader, False)
        scheduler.step()

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["valid_loss"].append(va_loss); history["valid_acc"].append(va_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Valid {va_loss:.4f}/{va_acc:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "acc": best_acc,
                "classes": class_names,
                "img_size": IMG_SIZE,
            }
            torch.save(best_state, MODEL_OUT)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val acc: {best_acc:.4f}")
                break

    print(f"Done in {(time.time()-t0)/60:.1f} min. Best acc: {best_acc:.4f}")
    print(f"Saved: {MODEL_OUT}")

    # ---- Evaluate ----
    ckpt = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels.numpy())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(4.5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Valid)")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
