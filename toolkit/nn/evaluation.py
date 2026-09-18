import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, input_tensor, output_tensor, target_tensor, data_range=2.0):
        # Detach tensors to prevent gradient tracking and memory leaks
        self.input = input_tensor.detach()
        self.output = output_tensor.detach()
        self.target = target_tensor.detach()

        # 1. Standard L1 Loss
        self.l1 = torch.mean(torch.abs(self.output - self.target)).detach()

        # 2. Continuous Percentage Accuracy
        abs_error = torch.abs(self.output - self.target)
        mean_relative_error = torch.mean(abs_error) / data_range
        self.accuracy = (1.0 - mean_relative_error).detach() * 100.0

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
