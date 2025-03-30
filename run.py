import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DogDataset(Dataset):
    def __init__(self, root_dir, df=None, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        if mode == 'train':
            self.ids = df['id'].values
            self.labels = df['breed'].values
        else:
            self.ids = [f.split('.')[0] for f in os.listdir(root_dir)]
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        if self.mode == 'train':
            label = self.labels[idx]
            label = breed_to_num[label]
        else:
            label = -1

        img_path = os.path.join(self.root_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    labels_df = pd.read_csv('labels.csv')
    breeds = labels_df['breed'].unique().tolist()
    global breed_to_num, num_to_breed
    breed_to_num = {breed:i for i, breed in enumerate(breeds)}
    num_to_breed = {i:breed for i, breed in enumerate(breeds)}

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DogDataset(root_dir='train', df=labels_df, transform=train_transform, mode='train')
    test_dataset = DogDataset(root_dir='test', transform=test_transform, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载预训练模型并冻结指定层
    model = models.resnet50(pretrained=True)
    freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2']
    for name, param in model.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False

    # 替换最后一层全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(breeds))
    model = model.to(device)

    # 重新定义优化器（仅优化未冻结的参数）
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        momentum=0.9
    )

    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 保存模型（可选）
    torch.save(model.state_dict(), 'model_frozen.pth')

if __name__ == "__main__":
    main()