import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class SnowDataset(Dataset):
    def __init__(self, snow_dir, mask_dir, transform=None):
        self.snow_dir = snow_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.snow_images = os.listdir(snow_dir)

    def __len__(self):
        return len(self.snow_images)

    def __getitem__(self, idx):
        snow_path = os.path.join(self.snow_dir, self.snow_images[idx])
        mask_path = os.path.join(self.mask_dir, self.snow_images[idx])
        snow_image = Image.open(snow_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")

        if self.transform:
            snow_image = self.transform(snow_image)
            mask_image = self.transform(mask_image)

        return snow_image, mask_image


def get_dataloaders(train_snow, train_mask, val_snow, val_mask, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = SnowDataset(train_snow, train_mask, transform=transform)
    val_dataset = SnowDataset(val_snow, val_mask, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
