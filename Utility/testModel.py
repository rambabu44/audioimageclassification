import torch
import numpy as np
import librosa
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import os

# Define the model architecture again (as it was during training)
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
        return self.fc(x)

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechClassifier(input_size=58, num_classes=10)  # Use 58 features and 10 classes as per your setup
model.load_state_dict(torch.load("models/speech_disorder_model.pth"))
model.to(device)
model.eval()  # Set model to evaluation mode

# Feature extraction function
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        # Ensure consistent feature size
        mfccs_mean = np.mean(mfccs.T, axis=0) if mfccs.shape[1] == 40 else np.zeros(40)
        chroma_mean = np.mean(chroma.T, axis=0) if chroma.shape[0] == 12 else np.zeros(12)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0) if spectral_contrast.shape[0] == 6 else np.zeros(6)
        
        return np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(58)  # Default zero vector if an error occurs

# Example usage for a new audio file
audio_file = "test_second.wav"  # Replace with your audio file path
features = extract_features(audio_file)
features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

# Perform the classification
with torch.no_grad():
    outputs = model(features)  # Get model output
    _, predicted_class = torch.max(outputs, 1)  # Get the class with the highest probability

# List of class labels (same as during training)
data_dir = "Data/genres_original"
labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# Get the predicted label
predicted_label = labels[predicted_class.item()]
print(f"Predicted Label: {predicted_label}")
