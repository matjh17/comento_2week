import os

# 폴더 설정
base_dir = "cifar_dogs_cats"
splits = ["train", "valid"]
class_map = {"cat": 0, "dog": 1}

for split in splits:
    for class_name, class_idx in class_map.items():
        image_dir = os.path.join(base_dir, split, class_name)
        label_dir = os.path.join(base_dir, split, "labels")

        os.makedirs(label_dir, exist_ok=True)

        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg"):
                name_only = os.path.splitext(filename)[0]
                label_path = os.path.join(label_dir, name_only + ".txt")

                # YOLO 라벨: 전체 이미지 중심에 객체가 있는 것으로 가정
                with open(label_path, "w") as f:
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
