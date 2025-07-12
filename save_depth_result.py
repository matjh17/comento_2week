import cv2
import numpy as np
import os

# 결과 저장 폴더 생성
os.makedirs("outputs", exist_ok=True)

# 이미지 로드
image = cv2.imread("sample.jpg")
if image is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Depth Map 생성 (의사 색상 적용)
depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# 3D 포인트 클라우드 계산
h, w = depth_map.shape[:2]
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = gray.astype(np.float32)
points_3d = np.dstack((X, Y, Z))  # (H, W, 3)

# 결과 시각화 이미지 저장
cv2.imwrite("outputs/original_image.jpg", image)
cv2.imwrite("outputs/depth_map.jpg", depth_map)

print("✅ 결과 이미지가 outputs/ 폴더에 저장되었습니다.")
