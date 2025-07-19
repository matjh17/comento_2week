from ultralytics import YOLO

# 사전 학습된 YOLOv8n 모델 불러오기
model = YOLO("yolov8n.pt")

# 사용자 데이터셋으로 학습하기
model.train(data="data.yaml", epochs=10, imgsz=64, name="cifar_yolo")
