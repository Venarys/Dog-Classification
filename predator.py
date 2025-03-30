import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载类别映射
def load_labels():
    labels_df = pd.read_csv('labels.csv')
    breeds = labels_df['breed'].unique().tolist()
    breed_to_num = {breed: i for i, breed in enumerate(breeds)}
    num_to_breed = {i: breed for i, breed in enumerate(breeds)}
    return num_to_breed

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 添加 batch 维度

# 加载模型
def load_model(num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load('model_frozen.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_tensor, num_to_breed):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = preds.item()
    return num_to_breed[predicted_class]

def main():
    # 检查输入图片是否存在
    image_path = 'test_image.jpg'  # 替换为你的图片文件名
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found!")
        return

    # 加载标签和模型
    num_to_breed = load_labels()
    num_classes = len(num_to_breed)
    model = load_model(num_classes)

    # 预处理图片并预测
    image_tensor = preprocess_image(image_path)
    predicted_breed = predict_image(model, image_tensor, num_to_breed)

    # 输出结果
    print(f"Predicted breed: {predicted_breed}")

if __name__ == "__main__":
    main()