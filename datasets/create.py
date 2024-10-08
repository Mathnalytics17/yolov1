from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotationJitter, RandomBlur
from datasets import YoloDataset

def create_dataloaders(img_list_path, train_ratio, valid_ratio, batch_size, input_size, S, B, num_classes):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(0.2, 0.5, 0.7, 0.07),
        transforms.RandomAdjustSharpness(3, p=0.2),
        RandomBlur(kernel_size=[3,3], sigma=[0.1, 2], p=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor()
    ])
    img_box_transform = [
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.05),
        RandomRotationJitter()
    ]

    # create yolo dataset
    dataset = YoloDataset(img_list_path, S, B, num_classes, transforms=transform, img_box_transforms=img_box_transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size

    # split dataset to train set, val set and test set three parts
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    valid_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)



    return train_loader, valid_loader
