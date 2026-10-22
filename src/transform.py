import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224 

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

Val_transform = A.Compose([
    A.Resize(IMG_SIZE,IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])