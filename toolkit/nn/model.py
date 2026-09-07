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

