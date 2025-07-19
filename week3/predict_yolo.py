import cv2
from ultralytics import YOLO

# 1. 학습된 YOLO 모델 로드
model_path = "C:/Users/lee/comento_2week/runs/detect/cifar_yolo/weights/best.pt"
model = YOLO(model_path)

# 2. 테스트 이미지 경로
image_path = "C:/Users/lee/comento_2week/week3/cifar_dogs_cats/valid/cat/cat_1.jpg"
image = cv2.imread(image_path)

# 3. 객체 탐지 실행
results = model(image)

# 4. 탐지 결과 시각화
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

# 5. 결과 출력
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1. 평가 수행
metrics = model.val()

# 평가 결과 출력 (괄호 제거!)
print("📌 Mean Precision:", metrics.box.mp)       # mean precision
print("📌 Mean Recall:", metrics.box.mr)          # mean recall
print("📌 mAP@0.5:", metrics.box.map50)           # mAP at IoU=0.5
print("📌 mAP@0.5:0.95:", metrics.box.map)        # mAP at IoU=0.5~0.95
