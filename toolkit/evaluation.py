import torch

class Evaluation:
    def __init__(self, input_tensor, output_tensor, target_tensor):
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
        # Avoids division by zero by adding a small epsilon
        tensor_range = 2.0

        # Calculate mean error relative to the total possible range rather than dividing by near-zero targets
        mean_relative_error = torch.mean(abs_error) / tensor_range

        # Continuous score: 0 error = 100% accuracy, maximum possible error (span of 2) = 0% accuracy
        self.accuracy = (1.0 - mean_relative_error).item() * 100.0
