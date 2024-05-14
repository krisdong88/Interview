from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# 加载CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 下载图像
url = "https://images.unsplash.com/photo-1517423440428-a5a00ad493e8"
image = Image.open(requests.get(url, stream=True).raw)

# 定义文本描述
texts = ["a photo of a black dog", "a photo of a white dog","a photo of a white cat"]

# 预处理图像和文本
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 获取模型输出
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 图像相对于文本的相似度
logits_per_text = outputs.logits_per_text    # 文本相对于图像的相似度
 
# 打印相似度
print(logits_per_image)
print(logits_per_text)
