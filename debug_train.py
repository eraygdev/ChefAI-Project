import torch
from torch.utils.data import DataLoader

from src.models.config import CLASS_TO_IDX, PROCESSED_DIR
from src.models.dataset import FoodDataset, get_eval_transforms
from src.models.model import build_model

model = build_model(pretrained=True, num_classes=30)

# Hangi head?
print("Head type:", type(model.get_classifier()).__name__)
print("Head params:", sum(p.numel() for p in model.get_classifier().parameters()))

# Backbone frozen, head trainable
for p in model.parameters():
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True

print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Data (aug YOK — saf öğrenme testi)
ds = FoodDataset(PROCESSED_DIR, "train", CLASS_TO_IDX, get_eval_transforms())
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
print("Train size:", len(ds))

# Kısa head-only training, lr=1e-3 (yüksek)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
crit = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).train()

for step, (imgs, lbls) in enumerate(loader):
    imgs, lbls = imgs.to(device), lbls.to(device)
    opt.zero_grad()
    loss = crit(model(imgs), lbls)
    loss.backward()
    opt.step()
    if step % 20 == 0:
        print(f"step {step:3d}  loss={loss.item():.3f}")
    if step >= 100:
        break
