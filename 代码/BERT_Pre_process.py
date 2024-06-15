from transformers import BertTokenizer

# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 示例文本
text = "Hello, how are you? I am using BERT tokenizer."

# 文本清理（示例）
text = text.lower()  # 转换为小写

# 分词和编码
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

print(encoded_input)
