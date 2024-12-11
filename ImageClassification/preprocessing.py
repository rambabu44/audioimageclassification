import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class Preprocessing:
    def __init__(self, image_size=(224, 224), batch_size=32, test_split=0.2, num_workers=4):
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def prepare_data(self, data_dir):
        data_transforms = self.get_transforms()
        dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
        class_names = dataset.classes

        # Split dataset
        train_size = int((1 - self.test_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, class_names
