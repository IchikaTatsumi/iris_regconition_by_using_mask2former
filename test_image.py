import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.mask2former import create_model
import json
import matplotlib.pyplot as plt

# 1. Cấu hình
CONFIG_PATH = 'configs/mask2former_config_kaggle.json'
CHECKPOINT_PATH = 'checkpoints/best_checkpoint.pth'
IMAGE_PATH = 'test_eye.jpg' # Thay bằng ảnh mắt bất kỳ

# 2. Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(CONFIG_PATH) as f: cfg = json.load(f)
model_cfg = cfg['model']
# Xóa key thừa
for k in ['architecture', 'model_type', 'use_checkpoint']: 
    if k in model_cfg: del model_cfg[k]

model = create_model(model_type="enhanced", **model_cfg)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.to(device).eval()

# 3. Xử lý ảnh
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
orig_size = img.shape[:2]

# Resize về 320x320 (Chuẩn training)
transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
x = transform(image=img)['image'].unsqueeze(0).to(device)

# 4. Dự đoán
with torch.no_grad():
    out = model(x)
    logits = out['logits'] if 'logits' in out else out['pred_masks']
    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy() # Lớp Iris

# 5. Hiển thị
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Ảnh gốc")
plt.subplot(1, 2, 2); plt.imshow(prob, cmap='jet', vmin=0, vmax=1); plt.title(f"Heatmap (Max: {prob.max():.2f})")
plt.colorbar()
plt.show()