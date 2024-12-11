import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset
data_dir = "numbers"

# Transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Split dataset into train and val sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))  # Update the final layer
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, criterion, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Training complete")
    return model

# Train the model
trained_model = train_model(model, criterion, optimizer, num_epochs=1)

# Save the model
torch.save(trained_model.state_dict(), "image_classification_model.pth")
























# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# import os

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Paths
# data_dir = "numbers"

# # Transformations
# data_transforms = {
#     "train": transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     "val": transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
# }

# # Datasets
# train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=data_transforms["train"])
# val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=data_transforms["val"])

# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# # Model
# model = models.resnet18(pretrained=True)  # Use a pretrained ResNet18
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Adjust for the number of classes
# model = model.to(device)

# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
# def train_model(model, criterion, optimizer, num_epochs=1):
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print("-" * 20)
        
#         # Each epoch has a training and validation phase
#         for phase in ["train", "val"]:
#             if phase == "train":
#                 model.train()
#                 dataloader = train_loader
#             else:
#                 model.eval()
#                 dataloader = val_loader

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()

#                 # Forward
#                 with torch.set_grad_enabled(phase == "train"):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # Backward + Optimize only if in training phase
#                     if phase == "train":
#                         loss.backward()
#                         optimizer.step()

#                 # Statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(dataloader.dataset)
#             epoch_acc = running_corrects.double() / len(dataloader.dataset)

#             print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

#     print("Training complete")
#     return model

# # Train the model
# trained_model = train_model(model, criterion, optimizer, num_epochs=1)

# # Save the model
# torch.save(trained_model.state_dict(), "image_classification_model.pth")
