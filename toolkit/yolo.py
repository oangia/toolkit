import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from toolkit.fpn import FPN, FPNLoss
import cv2
from google.colab.patches import cv2_imshow

class YOLO(inn.Model):
    def __init__(self):
        super().__init__()

    def fit(self, dataset = None, epochs=2000):
        self.dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        self.model = FPN().to(self.device)
        self.criterion = FPNLoss(box_weight=15.0)

        optimizer = optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

        print("--- Training Cooperative FPN Multi-Scale Model with Focal Loss ---")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets = tuple(t.to(self.device) for t in targets)

                optimizer.zero_grad()
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/1000 | Loss: {total_loss / len(self.dataloader):.4f}")

    def test(self, folder=""):
        # --- Inference and Evaluation Loop ---
        self.model.eval()
        for img_name in sorted(os.listdir(folder)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            test_img_path = os.path.join(folder, img_name)
            orig_image = cv2.imread(test_img_path)
            if orig_image is None:
                continue
            orig_h, orig_w = orig_image.shape[:2]

            input_img = cv2.resize(orig_image, (256, 256))
            input_tensor = torch.from_numpy(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)) \
                .float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            input_tensor = input_tensor * 2.0 - 1.0

            with torch.no_grad():
                pred_small, pred_medium, pred_large = self.model(input_tensor)

            candidates = []
            for preds, grid_size in zip([pred_small, pred_medium, pred_large], [32, 16, 8]):
                p = preds[0]
                obj_scores = torch.sigmoid(p[..., 0])
                gy, gx = torch.unravel_index(obj_scores.argmax(), obj_scores.shape)
                gy, gx = gy.item(), gx.item()

                conf = obj_scores[gy, gx].item()
                if conf < 0.7:  # Raised threshold to strictly filter out false positives
                    continue

                cell_pred = p[gy, gx]
                dx, dy, dw, dh = torch.sigmoid(cell_pred[1:]).tolist()

                cx = (gx + dx) / grid_size
                cy = (gy + dy) / grid_size
                candidates.append((conf, cx, cy, dw, dh))

            if len(candidates) > 0:
                best_candidate = max(candidates, key=lambda x: x[0])
                obj_conf, pred_cx, pred_cy, pred_w, pred_h = best_candidate

                xmin = int((pred_cx - pred_w / 2) * orig_w)
                ymin = int((pred_cy - pred_h / 2) * orig_h)
                xmax = int((pred_cx + pred_w / 2) * orig_w)
                ymax = int((pred_cy + pred_h / 2) * orig_h)

                center_x = int(pred_cx * orig_w)
                center_y = int(pred_cy * orig_h)

                print(f"\nInference Results for {img_name}:")
                print(f"Objectness Confidence: {obj_conf:.4f}")
                print(f"Predicted Box (Original Scale): [{xmin}, {ymin}, {xmax}, {ymax}]")

                cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.circle(orig_image, (center_x, center_y), 6, (0, 0, 255), -1)
                cv2.putText(orig_image, f"Obj: {obj_conf:.2f}", (xmin, max(30, ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                display_image = cv2.resize(orig_image, (512, 512))
                cv2_imshow(display_image)
            else:
                print(f"\nNo confident object detected for {img_name}")
