import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, image_size):
        self.folder_path = folder_path
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2.) - 1)
        ])
        self.image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

def get_dataloader(folder_path, batch_size, image_size):
    dataset = CustomImageDataset(folder_path, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)