import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Custom Dataset class
class SpeechDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        features = self.extract_features(file_path)
        if features.shape[0] != 58:
            print(f"Skipping {file_path} due to mismatched features")
            return None  # Handle gracefully
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

            mfccs_mean = np.mean(mfccs.T, axis=0) if mfccs.shape[1] == 40 else np.zeros(40)
            chroma_mean = np.mean(chroma.T, axis=0) if chroma.shape[0] == 12 else np.zeros(12)
            spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0) if spectral_contrast.shape[0] == 6 else np.zeros(6)
            # Ensure feature dimension consistency
            return np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return np.zeros(58)

# Neural Network Model
class SpeechClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SpeechClassifier, self).__init__()
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
        x = x[:, :58]
        return self.fc(x)

# Main
if __name__ == "__main__":
    # Paths and labels
    data_dir = "Data/genres_original"
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    file_paths, file_labels = [], []

    # Load file paths and labels
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for file in os.listdir(folder_path):
            file_paths.append(os.path.join(folder_path, file))
            file_labels.append(label)
    valid_data = [(f, l) for f, l in zip(file_paths, file_labels) if f is not None]
    file_paths, file_labels = zip(*valid_data)
    # Split data into train and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, file_labels, test_size=0.2, random_state=42
    )
    # Create Datasets and DataLoaders
    batch_size = 32
    train_dataset = SpeechDataset(train_paths, train_labels)
    test_dataset = SpeechDataset(test_paths, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 58  # Feature vector size
    num_classes = len(labels)
    model = SpeechClassifier(input_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for features, targets in train_loader:
        print(f"Feature shape: {features.shape}")  # Should be (batch_size, 58)
        break
    # Training loop
    num_epochs = 20
    print("Starting training on", device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Save model
    model_path = "speech_disorder_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
