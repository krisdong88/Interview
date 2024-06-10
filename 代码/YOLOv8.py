import cv2
from ultralytics import YOLO

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')  # 使用nano版本，可以根据需要选择其他版本如 yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧传入模型进行检测
    results = model(frame)

    # 处理结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = box.cls[0]
            label = model.names[int(cls)]
            
            # 绘制边界框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 显示结果
    cv2.imshow('YOLOv8 Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
