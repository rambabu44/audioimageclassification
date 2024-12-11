import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.cuda.amp import autocast, GradScaler


class ImageClassifier:
    def __init__(self, num_classes, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_model(num_classes)
        self.best_model_weights = None
        self.scalar = GradScaler()

    def initialize_model(self, num_classes):
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model.to(self.device)

    def get_optimizer(self, learning_rate=0.001):
        return optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_scheduler(self,optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001, early_stopping_patience = None):
        optimizer = self.get_optimizer(learning_rate)
        criterion = self.get_criterion()
        scheduler = self.get_scheduler(optimizer)

        best_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 20)

            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                    dataloader = train_loader
                else:
                    self.model.eval()
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                for i,(inputs, labels) in enumerate(dataloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            self.scalar.scale(loss).backward()
                            self.scalar.step(optimizer)
                            self.scalar.update()
                            # loss.backward()
                            # optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if i%10 == 0:
                        print(f"[{phase.capitalize()} Batch {i}] Loss: {loss.item():.4f}")

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        self.best_model_weights = self.model.state_dict()
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        return
            scheduler.step()
        self.model.load_state_dict(self.best_model_weights)
        print("Training complete. Best validation accuracy: {:.4f}".format(best_acc))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
