import os
import torchvision
from torchvision import transforms
from PIL import Image

# 폴더 구조 설정
base_dir = "cifar_dogs_cats"
splits = ["train", "valid"]
labels_map = {3: "cat", 5: "dog"}

# 1. 폴더 만들기
for split in splits:
    for label_name in ["cat", "dog"]:
        path = os.path.join(base_dir, split, label_name)
        os.makedirs(path, exist_ok=True)

# 2. CIFAR-10 불러오기

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

# 3. 필터링하고 저장 함수
def save_filtered(dataset, split):
    counter = {"cat": 0, "dog": 0}
    for img, label in dataset:
        if label in [3, 5]:  # cat or dog
            label_name = labels_map[label]
            counter[label_name] += 1
            img_pil = img
            save_path = os.path.join(base_dir, split, label_name, f"{label_name}_{counter[label_name]}.jpg")
            img_pil.save(save_path)
    print(f"✅ {split} 저장 완료: {counter}")

# 4. 저장 실행
save_filtered(trainset, "train")
save_filtered(testset, "valid")
