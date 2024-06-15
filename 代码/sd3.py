import torch
import clip
from PIL import Image

# 加载CLIP模型和预处理函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载并预处理图像
image_path = "C:/Users\krisd\Desktop\Github\Interview\笔记\zac.png"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 不同的文本描述
texts = ["a picture of a cat", "a picture of a dog", "an orange cat", "a cute orange cat", "a cat with orange fur"]
text_features_list = [model.encode_text(clip.tokenize([t]).to(device)) for t in texts]

# 提取图像特征
with torch.no_grad():
    image_features = model.encode_image(image)

# 计算并打印每个文本描述的相似度
for i, text_features in enumerate(text_features_list):
    cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    print(f"Cosine similarity with '{texts[i]}':", cosine_similarity.item())
