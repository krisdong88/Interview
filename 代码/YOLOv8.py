{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: ultralytics in c:\\users\\krisd\\anaconda3\\lib\\site-packages (8.2.15)\n",
      "Requirement already satisfied: opencv-python in c:\\users\\krisd\\anaconda3\\lib\\site-packages (4.9.0.80)\n",
      "Requirement already satisfied: matplotlib>=3.3.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (3.8.0)\n",
      "Requirement already satisfied: pillow>=7.1.2 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (10.2.0)\n",
      "Requirement already satisfied: pyyaml>=5.3.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (6.0.1)\n",
      "Requirement already satisfied: requests>=2.23.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (2.31.0)\n",
      "Requirement already satisfied: scipy>=1.4.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (1.10.1)\n",
      "Requirement already satisfied: torch>=1.8.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (2.3.0)\n",
      "Requirement already satisfied: torchvision>=0.9.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (0.18.0)\n",
      "Requirement already satisfied: tqdm>=4.64.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (4.65.0)\n",
      "Requirement already satisfied: psutil in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (5.9.0)\n",
      "Requirement already satisfied: py-cpuinfo in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (9.0.0)\n",
      "Requirement already satisfied: thop>=0.1.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (0.1.1.post2209072238)\n",
      "Requirement already satisfied: pandas>=1.1.4 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (2.1.4)\n",
      "Requirement already satisfied: seaborn>=0.11.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from ultralytics) (0.12.2)\n",
      "Requirement already satisfied: numpy>=1.21.2 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from opencv-python) (1.24.3)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from matplotlib>=3.3.0->ultralytics) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from matplotlib>=3.3.0->ultralytics) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from matplotlib>=3.3.0->ultralytics) (4.25.0)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from matplotlib>=3.3.0->ultralytics) (1.4.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from matplotlib>=3.3.0->ultralytics) (23.1)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from matplotlib>=3.3.0->ultralytics) (3.0.9)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from matplotlib>=3.3.0->ultralytics) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from pandas>=1.1.4->ultralytics) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from pandas>=1.1.4->ultralytics) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from requests>=2.23.0->ultralytics) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from requests>=2.23.0->ultralytics) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from requests>=2.23.0->ultralytics) (2.0.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from requests>=2.23.0->ultralytics) (2024.2.2)\n",
      "Requirement already satisfied: filelock in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from torch>=1.8.0->ultralytics) (3.13.1)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from torch>=1.8.0->ultralytics) (4.9.0)\n",
      "Requirement already satisfied: sympy in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from torch>=1.8.0->ultralytics) (1.12)\n",
      "Requirement already satisfied: networkx in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from torch>=1.8.0->ultralytics) (3.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from torch>=1.8.0->ultralytics) (3.1.3)\n",
      "Requirement already satisfied: fsspec in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from torch>=1.8.0->ultralytics) (2023.10.0)\n",
      "Requirement already satisfied: mkl<=2021.4.0,>=2021.1.1 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from torch>=1.8.0->ultralytics) (2021.4.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from tqdm>=4.64.0->ultralytics) (0.4.6)\n",
      "Requirement already satisfied: intel-openmp==2021.* in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from mkl<=2021.4.0,>=2021.1.1->torch>=1.8.0->ultralytics) (2021.4.0)\n",
      "Requirement already satisfied: tbb==2021.* in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from mkl<=2021.4.0,>=2021.1.1->torch>=1.8.0->ultralytics) (2021.12.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.16.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.3)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\users\\krisd\\anaconda3\\lib\\site-packages (from sympy->torch>=1.8.0->ultralytics) (1.3.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install ultralytics opencv-python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "0: 480x640 1 person, 8.1ms\n",
      "Speed: 1.0ms preprocess, 8.1ms inference, 2.0ms postprocess per image at shape (1, 3, 480, 640)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x000001E1C94B9120>\n",
      "Traceback (most recent call last):\n",
      "  File \"c:\\Users\\krisd\\anaconda3\\Lib\\site-packages\\torch\\utils\\data\\dataloader.py\", line 1479, in __del__\n",
      "    self._shutdown_workers()\n",
      "  File \"c:\\Users\\krisd\\anaconda3\\Lib\\site-packages\\torch\\utils\\data\\dataloader.py\", line 1443, in _shutdown_workers\n",
      "    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)\n",
      "  File \"c:\\Users\\krisd\\anaconda3\\Lib\\multiprocessing\\process.py\", line 149, in join\n",
      "    res = self._popen.wait(timeout)\n",
      "          ^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"c:\\Users\\krisd\\anaconda3\\Lib\\multiprocessing\\popen_spawn_win32.py\", line 110, in wait\n",
      "    res = _winapi.WaitForSingleObject(int(self._handle), msecs)\n",
      "          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
      "KeyboardInterrupt: \n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "from ultralytics import YOLO\n",
    "\n",
    "# 加载YOLOv8模型（你需要使用一个预训练的人脸检测模型，或者你自己训练的人脸检测模型）\n",
    "model = YOLO('yolov8s.pt')  # 替换为你的人脸检测模型\n",
    "\n",
    "# 打开摄像头\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "while True:\n",
    "    # 读取摄像头帧\n",
    "    ret, frame = cap.read()\n",
    "    if not ret:\n",
    "        break\n",
    "\n",
    "    # 使用YOLOv8进行预测\n",
    "    results = model(frame)\n",
    "\n",
    "    # 获取预测结果并在图像上绘制框\n",
    "    for result in results:\n",
    "        boxes = result.boxes  # 获取所有预测的边界框\n",
    "        for box in boxes:\n",
    "            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 提取坐标\n",
    "            conf = box.conf[0]  # 置信度\n",
    "            label = result.names[box.cls[0]]  # 类别标签\n",
    "\n",
    "            # 只对置信度高的预测进行处理\n",
    "            if conf > 0.5:\n",
    "                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)\n",
    "                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)\n",
    "\n",
    "    # 显示结果帧\n",
    "    cv2.imshow('YOLOv8 Face Detection', frame)\n",
    "\n",
    "    # 按'q'键退出循环\n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "        break\n",
    "\n",
    "# 释放摄像头并关闭所有窗口\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
