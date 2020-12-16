import albumentations as A

train_transform = A.Compose([
    A.Resize(),
    A.Normalize(mean=, std=),
    A.HorizontalFlip(p=0.25),
    A.VerticalFlip(p=0.25),
    ToTensor()
])


valid_transform = A.Compose([
    A.Resize(),
    A.Normalize(mean= , std= ),
    ToTensor()
])