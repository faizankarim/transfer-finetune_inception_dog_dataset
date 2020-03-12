from sklearn.utils import class_weight
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import glob
import keras

PATH = "."

batch_size = 256
IMG_HEIGHT = IMG_WIDTH = 299
IMG_SIZE = 299
train_dir = os.path.join(PATH, 'train_augmented')
validation_dir = os.path.join(PATH, 'test')
total_train = len(glob.glob(train_dir + "/**/*.*", recursive=True))
total_validation = len(glob.glob(validation_dir + "/**/*.*", recursive=True))
train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
epochs = 15

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')
print("total_training_images: ", total_train)
print("total_validation_images: ", total_validation)
labels = train_data_gen.class_indices
CLASSES = len(labels)
print("total_classes: ", CLASSES)
class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train_data_gen.classes),
    train_data_gen.classes)
print("class_weights", class_weights)

sample_training_images, _ = next(train_data_gen)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("images.png")

print("plotting_images")
plotImages(sample_training_images[:5])

print("loading_base_model")
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')

base_model.trainable = False

print("base_model_summary")
# Let's take a look at the base model architecture
base_model.summary()

global_average_layer = keras.layers.GlobalAveragePooling2D()
Dense_layer_1 = keras.layers.Dense(1024, activation="relu")
Dense_layer_2 = keras.layers.Dense(512, activation="relu")
Dense_layer_3 = keras.layers.Dense(256, activation="relu")
prediction_layer = keras.layers.Dense(CLASSES, activation='softmax')

model = keras.Sequential([
    base_model,
    global_average_layer,
    Dense_layer_1,
    Dense_layer_2,
    Dense_layer_3,
    prediction_layer
])

print("compiling_model_for_transfer_learning")
base_learning_rate = 0.0001
model.compile(optimizer=keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()
from keras.callbacks import EarlyStopping,  ModelCheckpoint
earlystop = EarlyStopping(patience=3, verbose=1)
checkpoint = ModelCheckpoint("transfer_learning_weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True, verbose=1)
callbacks = [checkpoint, earlystop]
initial_epochs = epochs
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=initial_epochs,
    validation_data=val_data_gen,
    validation_steps=total_validation // batch_size,
    class_weight=class_weights,
    callbacks=callbacks
)
print("saving_weights")
model.save("transfer_learning_model.hdf5")
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('transfer_learning.png')

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 150

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

fine_tune_epochs = epochs
total_epochs = initial_epochs + fine_tune_epochs
earlystop = EarlyStopping(patience=3, verbose=1)
checkpoint = ModelCheckpoint("fine_tuning_weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True, verbose=1)
callbacks = [checkpoint, earlystop]
history_fine = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_data_gen,
    validation_steps=total_validation // batch_size,
    class_weight=class_weights,
    callbacks=callbacks
)
model.save("finetune_learning_model.hdf5")

acc += history_fine.history['acc']
val_acc += history_fine.history['val_acc']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs - 1, initial_epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs - 1, initial_epochs - 1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("fine_tuning.png")
