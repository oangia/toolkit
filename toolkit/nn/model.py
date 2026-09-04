import os
import torch
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def save_model(self, state_dict):
        torch.save(state_dict, self.save_path)

    def load_model(self):
        if os.path.exists(self.save_path):
            print(f"-> Found existing checkpoint at '{self.save_path}'. Loading generator...")
            self.model_state = torch.load(self.save_path, map_location=self.device)
        else:
            print("-> Starting generator training from scratch.")

    def plot(self, epochs, val_l1, val_acc):
        # --- PLOT LOSS & ACCURACY CHARTS ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss Chart
        axes[0].plot(epochs, val_l1, label='Val L1 Loss', color='red', linewidth=2)
        axes[0].set_title('Loss History')
        axes[0].set_xlabel('Epochs')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy Chart
        axes[1].plot(epochs, val_acc, label='Val Pixel Accuracy (%)', color='green', linewidth=2)
        axes[1].set_title('Validation Pixel Accuracy (<10% Error)')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
