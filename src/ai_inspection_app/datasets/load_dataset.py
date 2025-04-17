from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch

class PCBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list((self.root_dir / "images").glob("*.jpg"))
        self.labels = self._load_labels()

    def _load_labels(self):
        labels = {}
        label_dir = self.root_dir / "labels"
        for label_file in label_dir.glob("*.txt"):
            img_name = label_file.stem + ".jpg"
            with open(label_file, "r") as f:
                lines = f.readlines()
            labels[img_name] = [list(map(float, line.strip().split())) for line in lines]
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels.get(img_path.name, [])

        if self.transform:
            image = self.transform(image)

        target = {
            "boxes": torch.tensor([l[1:] for l in label], dtype=torch.float32),
            "labels": torch.tensor([int(l[0]) for l in label], dtype=torch.int64),
        }

        return image, target
