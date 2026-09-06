import os
import torch
import matplotlib.pyplot as plt

class Model:
    def __init__(self, generator_class):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator_class().to(self.device)
    
    def save_model(self, path=None):
        target_path = path if path is not None else self.save_path
        torch.save(self.generator.state_dict(), target_path)

    def load_model(self):
        if os.path.exists(self.save_path):
            print(f"-> Found existing checkpoint at '{self.save_path}'. Loading generator...")
            state = torch.load(self.save_path, map_location=self.device)
            self.generator.load_state_dict(state)
        else:
            print("-> Starting generator training from scratch.")

    def plot(self, epochs, train_l1, val_l1, train_acc, val_acc):
        # --- PLOT LOSS & ACCURACY CHARTS (TRAIN VS VAL) ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss Chart
        axes[0].plot(epochs, train_l1, label='Train L1 Loss', color='blue', linestyle='--', linewidth=2)
        axes[0].plot(epochs, val_l1, label='Val L1 Loss', color='red', linewidth=2)
        axes[0].set_title('Loss History (Train vs Val)')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy Chart
        axes[1].plot(epochs, train_acc, label='Train Pixel Accuracy (%)', color='orange', linestyle='--', linewidth=2)
        axes[1].plot(epochs, val_acc, label='Val Pixel Accuracy (%)', color='green', linewidth=2)
        axes[1].set_title('Accuracy History (Train vs Val)')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
