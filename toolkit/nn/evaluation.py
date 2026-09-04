import torch
import torch.nn.functional as F

class Evaluation:
    def __init__(self, input_tensor, output_tensor, target_tensor, data_range=2.0):
        # 1. Standard L1 Loss
        self.l1 = torch.mean(torch.abs(output_tensor - target_tensor)).item()

        # 2. Change-Only L1 Loss
        change_mask = (torch.abs(input_tensor - target_tensor) > 1e-4).float()
        abs_error = torch.abs(output_tensor - target_tensor)
        if change_mask.sum() > 0:
            self.change_l1 = ((abs_error * change_mask).sum() / change_mask.sum()).item()
        else:
            self.change_l1 = 0.0

        # 3. Continuous Percentage Accuracy (calculated via relative error magnitude)
        tensor_range = data_range
        mean_relative_error = torch.mean(abs_error) / tensor_range
        self.accuracy = (1.0 - mean_relative_error).item() * 100.0

        # 4. SSIM (Structural Similarity Index)
        self.ssim = self._compute_ssim(output_tensor, target_tensor, data_range=tensor_range)

    def _compute_ssim(self, img1, img2, data_range=2.0, window_size=11, window_sigma=1.5):
        # Handle shape safety (expects N, C, H, W)
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0).unsqueeze(0)
            img2 = img2.unsqueeze(0).unsqueeze(0)
        elif img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        channels = img1.shape[1]
        
        # Build 2D Gaussian window
        coords = torch.arange(window_size, dtype=img1.dtype, device=img1.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * window_sigma ** 2))
        g /= g.sum()
        window_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
        window = window_2d.repeat(channels, 1, 1, 1)

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        return ssim_map.mean().item()
