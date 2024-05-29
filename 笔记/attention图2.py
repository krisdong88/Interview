import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from transformers import ViTModel, ViTFeatureExtractor

# 加载预训练的ViT模型和特征提取器
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 加载和预处理图像
image_path = 'zac.png'
image = Image.open(image_path).convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")

# 获取模型输出和attention权重
outputs = model(**inputs, output_attentions=True)
attention = outputs.attentions[-1]  # 使用最后一层的attention权重
attention = attention.mean(1)  # 平均多个heads

# 检查attention的形状
print("Attention shape:", attention.shape)  # 形状应为 [batch_size, num_patches, num_patches]

# 获取注意力图
attention_map = attention[0].detach().numpy()

# 检查attention_map的形状
print("Attention map shape:", attention_map.shape)  # 形状应为 [num_patches, num_patches]

# 计算num_patches，并确保其为正方形
num_patches = attention_map.shape[-1]

# 处理注意力图不是正方形的情况
if num_patches != attention_map.shape[-2]:
    raise ValueError(f"The attention map is not square: shape {attention_map.shape}")

# 将注意力图重塑为正方形
side_length = int(num_patches**0.5)
attention_map = attention_map.reshape((side_length, side_length))

# 将注意力图放大到与原始图像相同的尺寸
attention_map_resized = zoom(attention_map, (image.size[1] / side_length, image.size[0] / side_length), order=1)

# 标准化attention_map
attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / (np.max(attention_map_resized) - np.min(attention_map_resized))

# 显示和叠加attention图
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(attention_map_resized, cmap='jet', alpha=0.5)  # 使用半透明的方式叠加
plt.axis('off')
plt.show()
