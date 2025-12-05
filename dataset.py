import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class XrayDataset(Dataset):
    def __init__(self, root_dir, train=True):
        """
        root_dir: e.g. 'data/hospital_1/train' or 'data/hospital_1/val'
        train: True for training data (we add augmentation)
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []  # 0 = NORMAL, 1 = PNEUMONIA

        normal_dir = os.path.join(root_dir, "NORMAL")
        pneu_dir = os.path.join(root_dir, "PNEUMONIA")

        # Read NORMAL images
        if os.path.isdir(normal_dir):
            for fname in os.listdir(normal_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(normal_dir, fname))
                    self.labels.append(0)

        # Read PNEUMONIA images
        if os.path.isdir(pneu_dir):
            for fname in os.listdir(pneu_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(pneu_dir, fname))
                    self.labels.append(1)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label
