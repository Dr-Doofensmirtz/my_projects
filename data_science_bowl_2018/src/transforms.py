import albumentations as A
from albumentations.pytorch import ToTensor

class transform:
    train_transform = A.Compose([
        A.Resize(128,128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        ToTensor()
    ])


    valid_transform = A.Compose([
        A.Resize(128,128),
        A.Normalize(mean= (0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
        ToTensor()
    ])