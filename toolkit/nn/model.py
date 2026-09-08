import os
import torch
import matplotlib.pyplot as plt

class Tracker:
    def __init__(self):
        self.history_epochs = []
        self.history_train_l1 = []
        self.history_val_l1 = []
        self.history_train_acc = []
        self.history_val_acc = []

    def update(self, epoch, train_l1, val_l1, train_acc, val_acc):
        self.history_epochs.append(epoch)
        self.history_train_l1.append(train_l1)
        self.history_val_l1.append(val_l1)
        self.history_train_acc.append(train_acc)
        self.history_val_acc.append(val_acc)

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss Chart
        axes[0].plot(self.history_epochs, self.history_train_l1, label='Train L1 Loss', color='blue', linestyle='--', linewidth=2)
        axes[0].plot(self.history_epochs, self.history_val_l1, label='Val L1 Loss', color='red', linewidth=2)
        axes[0].set_title('Loss History (Train vs Val)')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy Chart
        axes[1].plot(self.history_epochs, self.history_train_acc, label='Train Pixel Accuracy (%)', color='orange', linestyle='--', linewidth=2)
        axes[1].plot(self.history_epochs, self.history_val_acc, label='Val Pixel Accuracy (%)', color='green', linewidth=2)
        axes[1].set_title('Accuracy History (Train vs Val)')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        
class Model:
    def __init__(self, generator_class, save_path="", train_dataset=None, val_dataset=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator_class().to(self.device)
        self.tracker = Tracker()
        self.save_path = save_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.load_model()
    
    def save_model(self, path=None):
        target_path = path if path is not None else self.save_path
        torch.save(self.generator.state_dict(), target_path)

    def load_model(self, path=None):
        target_path = path if path is not None else self.save_path
        if os.path.exists(target_path):
            print(f"-> Found existing checkpoint at '{target_path}'. Loading generator...")
            state = torch.load(target_path, map_location=self.device)
            self.generator.load_state_dict(state)
        else:
            print("-> Starting generator training from scratch.")

    def test(self):
        val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)

        for idx, (val_inputs, val_targets) in enumerate(val_loader):
            val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)

            with torch.no_grad():
                val_outputs = self.generator(val_inputs)

            result = inn.Evaluation(val_inputs, val_outputs, val_targets)
            result.log()
            result.plot()
