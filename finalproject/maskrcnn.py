from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt

path2data = "./data/train2017"
path2json = "./data/annotations/instances_train2017.json"
coco_train = CocoDetection(root=path2data, annFile=path2json)
print(f"Number of samples: {len(coco_train)}")

img, target = coco_train[0]

