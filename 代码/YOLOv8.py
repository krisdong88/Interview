import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8s.pt')  # 确保此路径正确

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # 使用YOLOv8进行预测
    results = model(frame)

    # 获取预测结果并在图像上绘制框
    for result in results:
        boxes = result.boxes  # 获取所有预测的边界框
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 提取坐标
            conf = box.conf[0].item()  # 置信度，转为Python浮点数
            cls_id = int(box.cls[0].item())  # 类别索引，转为Python整数

            # 确保类别索引在names中
            if cls_id in result.names:
                label = result.names[cls_id]  # 类别标签
            else:
                label = f'Unknown ({cls_id})'  # 未知类别，调试用

            # 只对置信度高的预测进行处理
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果帧
    cv2.imshow('YOLOv8 Face Detection', frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
