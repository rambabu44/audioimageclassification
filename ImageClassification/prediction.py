import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn


class Predictor:
    def __init__(self, model_path, num_classes, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, num_classes)

    def load_model(self, model_path, num_classes):
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path, class_names):
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert("RGB")
        image = image_transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)

        return class_names[preds.item()]
