import torch
from ImageClassification.prediction import Predictor
from ImageClassification.preprocessing import Preprocessing
from ImageClassification.training import ImageClassifier
def load_and_prepare_data(data_dir, image_size=(224, 224), batch_size=32, test_split=0.2):
    """
    Load and preprocess the image dataset.
    """
    preprocessing = Preprocessing(image_size=image_size, batch_size=batch_size, test_split=test_split)
    train_loader, val_loader, class_names = preprocessing.prepare_data(data_dir)
    return train_loader, val_loader, class_names

def train_model(train_loader, val_loader, class_names, num_epochs=5, learning_rate=0.001, model_path="image_classification_model.pth"):
    """
    Train the image classification model and save it to the specified path.
    """
    num_classes = len(class_names)
    classifier = ImageClassifier(num_classes)
    classifier.train(train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    return model_path

def evaluate_model(model_path, val_loader, class_names):
    """
    Evaluate the trained model on the validation dataset.
    """
    num_classes = len(class_names)
    predictor = Predictor(model_path, num_classes)
    
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(predictor.device), labels.to(predictor.device)
        with torch.no_grad():
            outputs = predictor.model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

def predict_single_image(model_path, image_path, class_names):
    """
    Predict the class of a single image using the trained model.
    """
    num_classes = len(class_names)
    predictor = Predictor(model_path, num_classes)
    predicted_class = predictor.predict(image_path, class_names)
    print(f"Predicted Class: {predicted_class}")
    return predicted_class

# Functions for importing and using in other scripts
def run_pipeline(data_dir, test_image_path, model_save_path="image_classification_model.pth"):
    """
    Full pipeline function to load data, train the model, evaluate it, and predict a single image.
    """
    # Step 1: Load and Prepare Data
    train_loader, val_loader, class_names = load_and_prepare_data(data_dir)

    # Step 2: Train the Model
    train_model(train_loader, val_loader, class_names, model_path=model_save_path)

    # Step 3: Evaluate the Model
    evaluate_model(model_save_path, val_loader, class_names)

    # Step 4: Predict a Single Image
    predict_single_image(model_save_path, test_image_path, class_names)
























# from preprocessing import Preprocessing
# from training import ImageClassifier
# from prediction import Predictor

# def main():
#     # Dataset directory
#     data_dir = "numbers"

#     # Preprocessing
#     preprocessing = Preprocessing(image_size=(224, 224), batch_size=32)
#     train_loader, val_loader, class_names = preprocessing.prepare_data(data_dir)

#     # Training
#     num_classes = len(class_names)
#     classifier = ImageClassifier(num_classes)
#     classifier.train(train_loader, val_loader, num_epochs=5, learning_rate=0.001)

#     # Save model
#     model_path = "image_classification_model.pth"
#     classifier.save_model(model_path)

#     # Prediction
#     predictor = Predictor(model_path, num_classes)
#     test_image_path = "numbers/1/1_2.jpg"
#     predicted_class = predictor.predict(test_image_path, class_names)
#     print(f"Predicted Class: {predicted_class}")


# if __name__ == "__main__":
#     main()
