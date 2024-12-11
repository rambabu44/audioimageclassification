from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
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


# Path to saved model and class names
model_path = "models/image_classification_model.pth"
class_names = dataset.classes  # Assuming the dataset variable is still available

# Load the model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))  # Match the number of classes
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Transform for the input image
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict class for an input image
def predict_image(image_path, model, class_names):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    image = image_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move image to device
    image = image.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    # Get predicted class
    predicted_class = class_names[preds.item()]
    return predicted_class

# Test the function
image_path = "numbers/1/1_2.jpg"
predicted_class = predict_image(image_path, model, class_names)
print(f"Predicted Class: {predicted_class}")
