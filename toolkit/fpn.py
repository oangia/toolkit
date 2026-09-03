import torch
import torch.nn as nn

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.lat_large = nn.Conv2d(256, 64, 1)
        self.lat_medium = nn.Conv2d(128, 64, 1)
        self.lat_small = nn.Conv2d(64, 64, 1)

        self.head_small = nn.Conv2d(64, 5, 1)
        self.head_medium = nn.Conv2d(64, 5, 1)
        self.head_large = nn.Conv2d(64, 5, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        f_large = self.lat_large(e5)
        f_medium_up = nn.functional.interpolate(f_large, size=e4.shape[2:], mode='nearest')
        f_medium = self.lat_medium(e4) + f_medium_up
        f_small_up = nn.functional.interpolate(f_medium, size=e3.shape[2:], mode='nearest')
        f_small = self.lat_small(e3) + f_small_up

        out_small = self.head_small(f_small).permute(0, 2, 3, 1).contiguous()
        out_medium = self.head_medium(f_medium).permute(0, 2, 3, 1).contiguous()
        out_large = self.head_large(f_large).permute(0, 2, 3, 1).contiguous()

        return out_small, out_medium, out_large

class FPNLoss(nn.Module):
    def __init__(self, box_weight=15.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma

    def calc_loss(self, preds, targets):
        pred_logits = preds[..., 0]
        target_obj = targets[..., 0]

        pred_prob = torch.sigmoid(pred_logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_logits, target_obj, reduction='none')

        pos_mask = (target_obj == 1)
        neg_mask = (target_obj == 0)

        # Focal Loss weighting to heavily penalize high-confidence false positives
        pt = torch.where(pos_mask, pred_prob, 1 - pred_prob)
        focal_weight = torch.where(pos_mask, self.alpha, (1 - self.alpha)) * (1 - pt) ** self.gamma
        focal_bce = focal_weight * bce_loss

        pos_loss = focal_bce[pos_mask].sum() if pos_mask.sum() > 0 else torch.tensor(0.0, device=preds.device)
        neg_loss = focal_bce[neg_mask].sum() if neg_mask.sum() > 0 else torch.tensor(0.0, device=preds.device)

        num_pos = pos_mask.sum().clamp(min=1).float()
        num_neg = neg_mask.sum().clamp(min=1).float()

        obj_loss = (pos_loss / num_pos) + (neg_loss / num_neg)

        if pos_mask.sum() > 0:
            pred_boxes = torch.sigmoid(preds[..., 1:])
            target_boxes = targets[..., 1:]
            box_loss = self.mse(pred_boxes[pos_mask], target_boxes[pos_mask])
        else:
            box_loss = torch.tensor(0.0, device=preds.device)

        return obj_loss + (self.box_weight * box_loss)

    def forward(self, preds_tuple, targets_tuple):
        return self.calc_loss(preds_tuple[0], targets_tuple[0]) + \
               self.calc_loss(preds_tuple[1], targets_tuple[1]) + \
               self.calc_loss(preds_tuple[2], targets_tuple[2])
