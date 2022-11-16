import cv2
import matplotlib.pyplot as plt
import glob
import os
import shutil


base_path = "E:\Lucru\ANUL II MASTER\MLAV\ProiectDificil\Pandora_18k_Merged\\"

# source_folder = r"E:\demos\files\reports\\"
# destination_folder = r"E:\demos\files\account\\"

folders = [ f.path for f in os.scandir(base_path) if f.is_dir() ]

image_path_by_class = []
aux_str = []

for folder in folders:
    aux_str = folder + "//" + "*.jpg"
    image_path_by_class.append(glob.glob(aux_str))


test_img = cv2.imread(image_path_by_class[0][2])
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(test_img)

test_img = cv2.imread(image_path_by_class[0][10])
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(test_img)

# # print(img_dict.keys())
# # print(img_dict[1])
# # test_img = cv2.imread(image_path_by_class[3][1])
# # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# # plt.figure()
# # plt.imshow(test_img)

plt.show()


