import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 假设我们已经有一个预训练的 StyleGAN 模型
# 这里我们使用 StyleGAN2 作为示例
# 加载预训练模型
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512', pretrained=True, useGPU=True)

# 生成一个随机潜在向量
z = torch.randn(1, 512).to('cuda')

# 生成图像
with torch.no_grad():
    generated_image = model.test(z)

# 转换为 PIL 图像并显示
transform = transforms.ToPILImage()
image = transform(generated_image.squeeze().cpu())
plt.imshow(image)
plt.axis('off')
plt.show()
