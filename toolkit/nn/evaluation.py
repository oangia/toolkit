import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, input_tensor, output_tensor, target_tensor, data_range=2.0):
        # Ensure 4D shape safety (N, C, H, W)
        input_tensor = self._ensure_4d(input_tensor)
        output_tensor = self._ensure_4d(output_tensor)
        target_tensor = self._ensure_4d(target_tensor)

        self.input = input_tensor
        self.output = output_tensor
        self.target = target_tensor

        # 1. Standard L1 Loss
        self.l1 = torch.mean(torch.abs(output_tensor - target_tensor)).item()

        # 2. Continuous Percentage Accuracy (calculated via relative error magnitude)
        tensor_range = data_range
        abs_error = torch.abs(output_tensor - target_tensor)
        mean_relative_error = torch.mean(abs_error) / tensor_range
        self.accuracy = (1.0 - mean_relative_error).item() * 100.0

    def _ensure_4d(self, tensor):
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            return tensor.unsqueeze(0)
        return tensor

    def _tensor_to_numpy(self, tensor):
        tensor_cpu = tensor.squeeze(0).cpu() * 0.5 + 0.5
        tensor_cpu = torch.clamp(tensor_cpu, 0.0, 1.0)
        return tensor_cpu.permute(1, 2, 0).numpy()

    def log(self):
        print(
            f"L1: {self.l1:.4f} | "
            f"Acc: {self.accuracy:.2f}%"
        )

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(self._tensor_to_numpy(self.output))
        axes[0].set_title("Output")
        axes[0].axis('off')

        axes[1].imshow(self._tensor_to_numpy(self.target))
        axes[1].set_title("Target")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
