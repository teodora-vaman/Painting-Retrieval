import cv2
import matplotlib.pyplot as plt
import glob
import os
import shutil
import numpy as np

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


base_path = "E:\Lucru\ANUL II MASTER\MLAV\ProiectDificil\Pandora_18k_Merged\\"

folders = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
classes_names = [ f.name for f in os.scandir(base_path) if f.is_dir() ]
classes = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18"]


image_path_by_class = []
aux_str = []

for folder in folders:
    aux_str = folder + "//" + "*.jpg"
    image_path_by_class.append(glob.glob(aux_str))


# test_img = cv2.imread(image_path_by_class[0][2])
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.imshow(test_img)

hist = []
total = 0
for base_class in image_path_by_class:
    hist.append(len(base_class))
    total += len(base_class)

plt.figure()
# ax = fig.add_axes([0,0,1,1])
plt.bar(classes,hist)
plt.title("Number of images per class")
plt.show()

print(total)


TEST_SIZE = 0.1
BATCH_SIZE = 64
SEED = 42
data = 22
# generate indices: instead of the actual data we pass in integers instead
train_indices, test_indices, _, _ = train_test_split(
    range(len(data)),
    data.targets,
    stratify=data.targets,
    test_size=TEST_SIZE,
    random_state=SEED
)

# generate subset based on indices
train_split = Subset(data, train_indices)
test_split = Subset(data, test_indices)

# create batches
train_batches = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
test_batches = DataLoader(test_split, batch_size=BATCH_SIZE)



