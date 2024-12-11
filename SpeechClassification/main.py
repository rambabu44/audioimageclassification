import os
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing, AudioDataset
from training import AudioClassifier, Trainer
from prediction import Predictor
from torch.utils.data import DataLoader
import torch

# Setup paths and labels
def load_data(data_dir):
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    file_paths, file_labels = [], []

    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for file in os.listdir(folder_path):
            file_paths.append(os.path.join(folder_path, file))
            file_labels.append(label)
    return file_paths,file_labels,labels

def prepare_datasets(data_dir, test_size=0.2, batch_size=32):
    """
    Prepares the train and test datasets and their DataLoaders.
    """
    file_paths, file_labels, label_names = load_data(data_dir)

    # Split into train and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, file_labels, test_size=test_size, random_state=42
    )

    # Preprocessing and dataset creation
    preprocessing = Preprocessing()
    train_dataset = AudioDataset(train_paths, train_labels, preprocessing)
    test_dataset = AudioDataset(test_paths, test_labels, preprocessing)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_dataset.feature_dim
    num_classes = len(label_names)

    return train_loader, test_loader, preprocessing, input_size, num_classes, label_names


def train_model(train_loader, input_size, num_classes, device="cuda"):
    """
    Train the model using the given DataLoader and hyperparameters.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(input_size, num_classes).to(device)
    trainer = Trainer(model, device)

    # Train the model
    trainer.train(train_loader)

    return model, trainer


def evaluate_model(trainer, test_loader):
    """
    Evaluate the trained model on the test dataset.
    """
    trainer.evaluate(test_loader)


def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    torch.save(model.state_dict(), model_path)


def predict(model_path, test_file, preprocessing, input_size, num_classes, device="cuda"):
    """
    Perform prediction on a single audio file.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    predictor = Predictor(model_path, input_size, num_classes, device)

    features = preprocessing.extract_features(test_file)
    if features is None:
        raise ValueError(f"Failed to extract features from {test_file}")
    predicted_class_id = predictor.predict(features)
    predicted_label = preprocessing.inverse_transform_label(predicted_class_id)
    return predicted_label





# from main_pipeline import (
#     prepare_datasets,
#     train_model,
#     evaluate_model,
#     save_model,
#     predict,
# )

# # 1. Prepare datasets
# data_dir = "Data/genres_original"
# train_loader, test_loader, preprocessing, input_size, num_classes, label_names = prepare_datasets(data_dir)

# # 2. Train the model
# device = "cuda"
# model, trainer = train_model(train_loader, input_size, num_classes, device=device)

# # 3. Evaluate the model
# evaluate_model(trainer, test_loader)

# # 4. Save the model
# model_path = "audio_classifier.pth"
# save_model(model, model_path)

# # 5. Predict on a new audio file
# test_file = "test_audio.wav"
# predicted_label = predict(model_path, test_file, preprocessing, input_size, num_classes, device=device)
# print(f"Predicted Label: {predicted_label}")
