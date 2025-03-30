import os
import paddle
import pandas as pd
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize

# 预处理（修正 Transpose 参数）
transform = Compose([
    Resize((224, 224)),
    Transpose(order=[2, 0, 1]),  # 使用 order 参数
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义数据集类（保持不变）
class DogDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.df = pd.read_csv(label_file)
        self.transform = transform

        self.class_names = sorted(self.df['breed'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

    def __getitem__(self, index):
        img_id = self.df.iloc[index]['id']
        img_path = os.path.join(self.data_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片: {img_path}")

        img = paddle.vision.image_load(img_path)
        label = self.class_to_idx[self.df.iloc[index]['breed']]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)

# 使用相对路径
train_dir = "train"
test_dir = "test"
label_file = "labels.csv"

# 创建数据集
train_dataset = DogDataset(train_dir, label_file, transform=transform)
test_dataset = DogDataset(test_dir, label_file, transform=transform)

# 数据加载器
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=32, shuffle=False)

import paddle.vision.models as models

# 加载预训练 ResNet50
backbone = models.resnet50(pretrained=True)

# 冻结除最后一层外的所有层
for param in backbone.parameters():
    param.stop_gradient = True  # 冻结所有参数

# 解冻最后一层（关键修正）
for param in backbone.fc.parameters():
    param.stop_gradient = False  # 允许最后一层更新

# 替换全连接层（关键修正）
# 获取原始全连接层的输入维度
original_in_features = backbone.fc.weight.shape[1]  # 确保是 2048
num_classes = len(train_dataset.class_names)

# 替换为新的全连接层
backbone.fc = paddle.nn.Linear(
    in_features=original_in_features,
    out_features=num_classes
)

# 验证层参数（可选）
print(f"原始全连接层权重形状: {backbone.fc.weight.shape}")  # 应输出 (类别数, 2048)

# 封装模型
model = paddle.Model(backbone)

# 配置训练参数（优化器仅训练最后一层）
model.prepare(
    optimizer=paddle.optimizer.Adam(
        learning_rate=0.001,
        parameters=backbone.fc.parameters()  # 指定最后一层参数
    ),
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy()
)

# 训练
model.fit(
    train_data=train_loader,
    epochs=5,
    eval_data=test_loader  # 修正：直接使用 test_loader
)

# 评估
model.evaluate(test_loader)