import os
import glob
import cv2
import imutils

# STATIC
dir_read = 'Images'
train_dir = 'train/'
test_dir = 'test/'

split_ratio = 10
size_image = 300
j = 0
m = 0

root_paths = glob.glob(dir_read + '/*', recursive=True)

for p in root_paths:
    _, folder_name = p.split('/')
    if not os.path.exists(train_dir + folder_name):
        os.makedirs(train_dir + folder_name)
    if not os.path.exists(test_dir + folder_name):
        os.makedirs(test_dir + folder_name)
    img_paths = glob.glob(p + '/*.*', recursive=True)
    print(folder_name)
    for k in img_paths:
        print(k)
        image = cv2.imread(k)
        height, width, _ = image.shape
        if height > width:
            image = imutils.resize(image, width=size_image)
        else:
            image = imutils.resize(image, height=size_image)

        if m % split_ratio == 0:
            cv2.imwrite(test_dir + folder_name + "/" + str(m) + ".png", image)
        else:
            cv2.imwrite(train_dir + folder_name + "/" + str(m) + ".png", image)

        print(image.shape)
        cv2.imshow("image", image)
        cv2.waitKey(1)
        print(m)
        m = m + 1
    j = j + 1