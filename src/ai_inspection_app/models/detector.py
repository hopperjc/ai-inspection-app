import torch
from pathlib import Path
from PIL import Image
from ai_inspection_app.datasets.load_dataset import PCBDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO

def train_detector(epochs=10, batch_size=4):
    train_dataset = PCBDataset(
        root_dir="data/train",
        transform=T.ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = YOLO('yolov5s.pt')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]

            optimizer.zero_grad()
            outputs = model(images)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

    Path("weights").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "weights/yolov5s_inspection.pth")
    print("Modelo salvo em weights/yolov5s_inspection.pth")

def infer_detector(image_path):
    model = YOLO('yolov5s.pt')
    model.load_state_dict(torch.load("weights/yolov5s_inspection.pth"))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = T.ToTensor()(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs.xyxy[0].cpu().numpy()

    plt.imshow(np.array(image))
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
        plt.text(x1, y1, f"{int(cls)} {conf:.2f}", color="red")
    plt.show()
