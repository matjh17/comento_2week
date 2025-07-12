
## 🖼️ 주요 기능

### ✅ 1. 2D → 3D 변환

- 이미지 → Grayscale 변환
- `cv2.applyColorMap`을 이용한 Depth Map 생성
- `np.meshgrid`와 Z-depth값을 이용한 3D Point Cloud 생성

> 📂 관련 파일:
- `depth_utils.py` : 핵심 함수 정의  

---

### 🧪 2. Unit Test 코드 (pytest 기반)

#### 테스트 대상 함수
```python
# generate_depth_map(image) : 이미지 → Depth Map
# generate_point_cloud(grayscale_image) : 그레이스케일 → 3D 포인트 클라우드
테스트 코드 (test_3d_processing.py)
python
복사
편집
import numpy as np
import pytest
import cv2
from depth_utils import generate_depth_map, generate_point_cloud

def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = generate_depth_map(image)
    assert result.shape == image.shape
    assert isinstance(result, np.ndarray)

def test_generate_point_cloud():
    gray = np.ones((50, 50), dtype=np.uint8) * 128
    cloud = generate_point_cloud(gray)
    assert cloud.shape == (50, 50, 3)
    assert cloud.dtype == np.float32
실행 명령어
bash
복사
편집
pytest test_3d_processing.py
실행 결과
diff
복사
편집
============================= test session starts =============================
collected 2 items

test_3d_processing.py ..                                               [100%]

============================== 2 passed in 0.05s ==============================
