import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class Trainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, train_loader, num_epochs=20):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for features, targets in train_loader:
                features, targets = features.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
