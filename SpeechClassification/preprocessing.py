import librosa
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os

class Preprocessing:
    def __init__(self,n_mfcc = 40, include_chroma = True, include_spectral_contrast=True,feature_dim = None):
        self.n_mfcc = n_mfcc
        self.include_chroma = include_chroma
        self.include_spectral_contrast = include_spectral_contrast
        self.feature_dim = feature_dim
        self.label_encoder = LabelEncoder()

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None)
            features = []

            #MFCCs
            if self.n_mfcc:
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
                features.extend(np.mean(mfccs.T, axis=0))

            #Chroma
            if self.include_chroma:
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                features.extend(np.mean(chroma.T,axis=0))

            #Spectral contrast
            if self.include_spectral_contrast:
                spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
                features.extend(np.mean(spectral_contrast.T, axis=0))

            feature_vector = np.array(features)

            if self.feature_dim is not None:
                feature_vector = self._ensure_feature_dim(feature_vector)

            # mfccs_mean = np.mean(mfccs.T, axis=0) if mfccs.shape[1] == 40 else np.zeros(40)
            # chroma_mean = np.mean(chroma.T, axis=0) if chroma.shape[0] == 12 else np.zeros(12)
            # spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0) if spectral_contrast.shape[0] == 6 else np.zeros(6)

            return feature_vector
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return np.zeros(self.feature_dim)
    
    def _ensure_feature_dim(self,feature_vector):
        if len(feature_vector) > self.feature_dim:
            return feature_vector[:self.feature_dim]
        elif len(feature_vector) < self.feature_dim:
            padding = np.zeros(self.feature_dim - len(feature_vector))
            return np.hstack((feature_vector,padding))
        return feature_vector
    
        
    def fit_labels(self,labels):
        return self.label_encoder.fit_transform(labels)

    def inverse_transform_label(self,label_id):
        return self.label_encoder.inverse_transform([label_id])[0]

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, preprocessing: Preprocessing):
        feature_dim = None
        for file_path in file_paths:
            sample_features = preprocessing.extract_features(file_path)
            if sample_features is not None:
                feature_dim = len(sample_features)
                break
        if feature_dim is None:
            raise ValueError(target = {"message":"Failed to extract features from any audio files!"})
        preprocessing.feature_dim = feature_dim
        self.file_paths = file_paths
        self.labels = preprocessing.label_encoder.fit_transform(labels)
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        features = self.preprocessing.extract_features(file_path)
        if features in None:
            raise ValueError(target = {"message":"Invalid feature for file:{file_path}"})
        return features, label
