import numpy as np
import torch
from torchvision import models, transforms
from scipy.linalg import sqrtm
from PIL import Image

# 加载预训练的 InceptionV3 模型
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.fc = torch.nn.Identity()  # 去掉分类层，只输出 avgpool 层的特征
inception_model.eval()

# 图像预处理
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(299),  # InceptionV3 输入尺寸
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img)
    img = img.unsqueeze(0)  # 添加 batch 维度
    return img

# 计算 FID
def calculate_fid(real_images, generated_images):
    real_features = []
    generated_features = []

    # 提取真实图像和生成图像的 InceptionV3 特征
    with torch.no_grad():
        for img_path in real_images:
            img = preprocess_image(img_path)
            output = inception_model(img)  # 提取 avgpool 层特征
            real_features.append(output.cpu().numpy().squeeze())

        for img_path in generated_images:
            img = preprocess_image(img_path)
            output = inception_model(img)  # 提取 avgpool 层特征
            generated_features.append(output.cpu().numpy().squeeze())

    real_features = np.array(real_features)
    generated_features = np.array(generated_features)

    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # 计算协方差矩阵的平方根
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # 如果出现复数，则取实部

    # 计算 FID
    diff = mu_real - mu_gen
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return fid

# 测试
real_images = ["D:/Lesson 17/base/base/cmp_b0010.jpg","D:/Lesson 17/base/base/cmp_b0010.jpg","D:/Lesson 17/base/base/cmp_b0010.jpg"]  # 替换为你的真实图片路径
generated_images = ["D:/Lesson 17/output_image.jpg","D:/Lesson 17/output_image.jpg","D:/Lesson 17/output_image.jpg"]  # 替换为你的生成图片路径
fid_score = calculate_fid(real_images, generated_images)
print("FID Score: ", fid_score)
