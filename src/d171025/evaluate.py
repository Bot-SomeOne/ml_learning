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

# --- Reproducibility ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

# --- Datasets ---
train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tfms)
valid_ds = datasets.ImageFolder(str(VALID_DIR), transform=valid_tfms)

if len(train_ds.classes) == 0:
    print("No classes found in training data directory. Expected subfolders per class.")
    raise SystemExit(1)
if len(valid_ds) == 0:
    print("Validation dataset is empty. Nothing to evaluate.")
    raise SystemExit(0)

assert train_ds.classes == valid_ds.classes, "Train/valid class sets differ"
class_names = train_ds.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# --- DataLoader knobs for macOS/MPS ---
is_cuda = (device.type == "cuda")
is_mps  = (device.type == "mps")
NUM_WORKERS = 0 if not is_cuda else (os.cpu_count() or 4)
PIN_MEMORY  = True if is_cuda else False
PERSISTENT  = True if (is_cuda and NUM_WORKERS > 0) else False

# --- Loaders ---
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

# --- Build and load model ---
def build_model(num_classes: int) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT  # torchvision >= 0.13
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)  # fallback for older torchvision
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

model = build_model(num_classes)
if Path(MODEL_OUT).exists():
    try:
        state = torch.load(MODEL_OUT, map_location=device)
        if isinstance(state, nn.Module):
            model = state
            # ensure final layer matches num_classes
            try:
                if getattr(model.fc, "out_features", None) != num_classes:
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, num_classes)
            except Exception:
                pass
        elif isinstance(state, dict):
            # common cases: full state_dict or {'state_dict': ...}
            for key in ("state_dict", "model_state_dict", "weights"):
                if key in state:
                    state = state[key]
                    break
            model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {MODEL_OUT}")
    except Exception as e:
        print(f"Warning: could not load weights from {MODEL_OUT}: {e}")
else:
    print(f"Warning: model weights not found at {MODEL_OUT}. Using ImageNet-initialized model.")

model = model.to(device)
model.eval()

# --- Inference to collect predictions ---
all_labels, all_preds = [], []
with torch.no_grad():
    for xb, yb in valid_loader:
        xb = xb.to(device, non_blocking=is_cuda)
        yb = yb.to(device, non_blocking=is_cuda)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(yb.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

if len(all_labels) == 0:
    print("No predictions collected from the validation loader.")
    raise SystemExit(1)

# --- Metrics ---
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
labels_order = list(range(num_classes))

print("\n=== Classification Report ===")
print(classification_report(
    all_labels, all_preds,
    labels=labels_order,              # explicit labels fixes mismatch with target_names
    target_names=class_names,         # keep human-readable class names
    digits=4,
    zero_division=0
))
acc = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {acc:.4f}")

# --- Confusion matrix plot ---
cm = confusion_matrix(all_labels, all_preds, labels=labels_order)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Valid)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
plt.show()