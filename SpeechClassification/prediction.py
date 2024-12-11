import torch
import numpy as np
from technical.model.New.SpeechClassification.training import AudioClassifier

class Predictor:
    def __init__(self, model_path, input_size, num_classes, device):
        self.device = device
        self.model = AudioClassifier(input_size, num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def predict(self, features):
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(features)
            _, predicted_class = torch.max(outputs, 1)
        return predicted_class.item()
