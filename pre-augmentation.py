from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import glob
import os
import numpy as np

# IMAGE_WIDTH = 224
# IMAGE_HEIGHT = 224
# IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

datagen = ImageDataGenerator(
    # rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)


def get_pic(file):
    img = load_img(file)
    x = np.expand_dims(img, axis=0)
    return x

batch_size = 10
paths = ["train/**/*.*"]
j = 0
for p in paths:
    path = glob.glob(p, recursive=True)
    for p_img in path:
        train_it = None
        img = get_pic(p_img)
        folder_name, class_name, image_name = p_img.split('/')
        # batch_size = 3
        if not os.path.exists(folder_name + "_augmented/" + class_name + '/'):
            os.makedirs(folder_name + "_augmented/" + class_name + '/')
        train_it = datagen.flow(x=img, batch_size=1, save_to_dir=folder_name + "_augmented/" + class_name + '/')
        j = j + 1
        print("image num: ", j)
        for i in range(batch_size):
            print("aumented_image: ", i)
            batchX = train_it.next()
