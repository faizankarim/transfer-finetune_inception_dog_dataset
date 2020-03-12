import json
from keras.preprocessing.image import ImageDataGenerator
import os

PATH = "."

batch_size = 256
IMG_HEIGHT = IMG_WIDTH = 299
IMG_SIZE = 299
train_dir = os.path.join(PATH, 'train_augmented')
validation_dir = os.path.join(PATH, 'test')

train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
epochs = 15

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
print(train_data_gen.class_indices)
with open('labels.txt', 'w') as outfile:
    json.dump(train_data_gen.class_indices, outfile)