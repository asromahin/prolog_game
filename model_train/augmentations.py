import albumentations as albu
import random
from config import config
from torchvision import transforms

def get_training_augmentation():
    train_transform = [
        transforms.ToPILImage(mode="RGB"),
        transforms.Resize((config.size, config.size)),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                 transforms.RandomGrayscale(p=0.5)]),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]
    return transforms.Compose(train_transform)




def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        transforms.ToPILImage(mode="RGB"),
        transforms.Resize((config.size, config.size)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(test_transform)
