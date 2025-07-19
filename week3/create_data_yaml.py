yaml_content = """\
train: cifar_dogs_cats/train
val: cifar_dogs_cats/valid

nc: 2
names: ['cat', 'dog']
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("✅ data.yaml 생성 완료!")
