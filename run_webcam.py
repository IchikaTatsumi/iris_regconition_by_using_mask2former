import cv2
import torch
import numpy as np
import json
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
import torch.nn.functional as F

# --- SETUP ĐƯỜNG DẪN ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.models.mask2former import EnhancedMask2Former
except ImportError:
    print("❌ LỖI: Không tìm thấy module 'src'.")
    sys.exit(1)

class IrisSegmentor:
    def __init__(self, config_path, checkpoint_path):
        # Thiết bị
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"🚀 Phan cung: {device_name}")

        # 1. Load Config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Xử lý config
        model_cfg = self.config.get('model', self.config.get('model_config', {}))
        for k in ['architecture', 'model_type', 'use_checkpoint']:
            if k in model_cfg: del model_cfg[k]

        # 2. Init Model
        print("🏗️ Dang khoi tao Model...")
        try:
            self.model = EnhancedMask2Former(**model_cfg)
        except TypeError:
            model_cfg = {k:v for k,v in model_cfg.items() if k in ['num_labels', 'model_name', 'num_queries']}
            self.model = EnhancedMask2Former(**model_cfg)

        # 3. Load Weights
        print(f"⚖️ Dang tai trong so (Weights)...")
        if not os.path.exists(checkpoint_path):
            print(f"❌ File khong ton tai: {checkpoint_path}")
            sys.exit(1)

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt.get('model', ckpt)))
            self.model.load_state_dict(state_dict)
            print("✅ Da tai Model thanh cong!")
        except Exception as e:
            print(f"❌ Loi load weights: {e}")
            sys.exit(1)

        self.model.to(self.device).eval()

        # 4. Transform - QUAN TRỌNG: 320x320 để tối ưu tốc độ
        self.img_size = 320
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict_raw_probs(self, frame):
        """
        Trả về bản đồ xác suất kích thước nhỏ (320x320) để xử lý cho nhanh
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = self.transform(image=img_rgb)['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = self.model(x)
                logits = out.get('pred_masks', out.get('logits', out))
                
                # Softmax -> Xác suất
                probs = F.softmax(logits, dim=1)
                
                # Lấy kênh Iris (Class 1)
                iris_prob = probs[0, 1, :, :] # [320, 320]
                
        # Trả về numpy array kích thước 320x320 (Chưa resize vội)
        return iris_prob.cpu().float().numpy()

def nothing(x):
    pass

def main():
    CONFIG = 'configs/mask2former_config_kaggle.json'
    CKPT = 'checkpoints/best_checkpoint.pth'

    segmentor = IrisSegmentor(CONFIG, CKPT)
    
    cap = cv2.VideoCapture(0)
    # Thiết lập độ phân giải Webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- TẠO CỬA SỔ ĐIỀU KHIỂN TIẾNG VIỆT ---
    # Lưu ý: OpenCV đôi khi hiển thị tiếng Việt có dấu bị lỗi font trên Windows Title
    # nên mình dùng Tiếng Việt không dấu hoặc ASCII chuẩn để an toàn nhất.
    window_name = "Dieu Chinh Mong Mat (Iris Tuner)" 
    cv2.namedWindow(window_name)
    
    # 1. Thanh trượt độ nhạy: Mặc định 35%
    cv2.createTrackbar("Do Nhay %", window_name, 35, 100, nothing)
    # 2. Thanh trượt làm mịn: Mặc định 5
    cv2.createTrackbar("Lam Min", window_name, 5, 20, nothing)
    
    print("\n🟢 DANG CHAY... Chinh thanh truot de toi uu ket qua!")
    print("👉 Bam 'q' de thoat chuong trinh.")

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Lật ảnh gương cho tự nhiên
        frame = cv2.flip(frame, 1)
        original_h, original_w = frame.shape[:2]

        # 1. Lấy xác suất thô (Kích thước nhỏ 320x320) -> TỐI ƯU HÓA
        prob_map_small = segmentor.predict_raw_probs(frame)

        # 2. Lấy giá trị từ thanh trượt
        thresh_val = cv2.getTrackbarPos("Do Nhay %", window_name) / 100.0
        kernel_size = cv2.getTrackbarPos("Lam Min", window_name)
        if kernel_size < 1: kernel_size = 1

        # 3. Xử lý trên ảnh nhỏ (Nhanh hơn 4 lần so với xử lý ảnh to)
        # Tạo mask thô
        mask_small = (prob_map_small > thresh_val).astype(np.uint8)

        # Lọc nhiễu (Morphology)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)  # Xóa nhiễu hạt
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel) # Lấp lỗ hổng

        # 4. Chỉ giữ vùng lớn nhất (Loại bỏ rác)
        cnts, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask_small = np.zeros_like(mask_small)
        
        has_iris = False
        if cnts:
            largest_cnt = max(cnts, key=cv2.contourArea)
            # Chỉ vẽ nếu vùng đủ lớn (> 30 pixel ở độ phân giải thấp)
            if cv2.contourArea(largest_cnt) > 30:
                cv2.drawContours(clean_mask_small, [largest_cnt], -1, 1, -1)
                has_iris = True
        
        # 5. Phóng to Mask lên kích thước Webcam (Resize 1 lần duy nhất ở đây)
        # Dùng INTER_NEAREST (Nhanh nhất) hoặc INTER_LINEAR (Mượt hơn xíu)
        final_mask = cv2.resize(clean_mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # 6. Vẽ lên ảnh gốc
        result = frame.copy()
        
        if has_iris:
            # Tạo lớp phủ màu xanh
            overlay = np.zeros_like(frame)
            overlay[final_mask == 1] = (0, 255, 0) # Xanh lá
            
            # Blend vào ảnh gốc
            result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Vẽ viền bao quanh (Tìm lại contour trên mask lớn để vẽ viền cho nét)
            final_cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if final_cnts:
                largest_final_cnt = max(final_cnts, key=cv2.contourArea)
                cv2.drawContours(result, [largest_final_cnt], -1, (0, 255, 255), 2) # Viền vàng

        # Tính và hiển thị FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Hiển thị thông số tiếng Việt
        cv2.putText(result, f"Toc do: {int(fps)} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result, f"Nguong: {int(thresh_val*100)}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow(window_name, result)
        
        # Hiển thị Heatmap ở cửa sổ nhỏ (Debug)
        heatmap = (prob_map_small * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imshow("Ban Do Nhiet (Heatmap)", cv2.resize(heatmap_color, (320, 240)))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()