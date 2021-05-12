import math
import numpy as np
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import Input, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator

BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_SAMPLES = 10000
EPOCHS = 20
RESULTSPATH = "results"
class MobileNet():

    def __init__(self, train_path, val_path, width, height, augment=False, save_history=True):
        self.train_path = train_path
        self.val_path = val_path
        self.augment = augment
        self.img_width, self.img_height = width, height
        self.save_history = save_history
    
    def train(self, model_output):
        
        if self.augment:
            # source: Pracitical Deep Learning for Cloud, Mobile, & Edge book
            # TODO: data seems to have already been augmented, maybe try without augmentation
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                               rotation_range=20,
                                               zoom_range=0.2,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               horizontal_flip=True)
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = train_datagen.flow_from_directory(
                            self.train_path,
                            target_size=(self.img_height, self.img_width),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            seed=0,
                            class_mode="binary")
            
        validation_generator = val_datagen.flow_from_directory(
                                self.val_path,
                                target_size=(self.img_height, self.img_width),
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                class_mode="binary")

        model = self._get_model()

        print("Compiling MobileNetV2 classifier...")
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=LEARNING_RATE),
                      metrics=["accuracy"])
        
        print("\nTraining MobileNetV2 model...")
        num_samples = sum([len(files) for r, d, files in os.walk(self.train_path)])
        print("Training on {} samples".format(num_samples))

        num_steps = math.ceil(float(num_samples)/BATCH_SIZE)
        history = model.fit_generator(train_generator,
                            steps_per_epoch=num_steps,
                            epochs=EPOCHS,
                            validation_data=validation_generator,
                            validation_steps=num_steps)

        if self.save_history:
            plt.style.use("dark_background")
            plt.figure()

            plt.plot(np.arange(0, EPOCHS), history.history["loss"], label="train_loss")
            plt.plot(np.arange(0, EPOCHS), history.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, EPOCHS), history.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, EPOCHS), history.history["val_accuracy"], label="val_acc")
            plt.title("MobileNetV2 Training Loss & Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(RESULTSPATH, "MobileNetV2_epoch{}_batchsize{}_augment{}_trainval_results.png"
                                                   .format(EPOCHS, BATCH_SIZE, "YES" if self.augment else "NO")))

        if model_output:
            output_path = os.path.join(model_output, "MobileNetV2_epochs{}_batchsize{}_augment{}.h5"
                                       .format(EPOCHS, BATCH_SIZE, "YES" if self.augment else "NO"))
            print("Saving model to {}...".format(output_path))
            model.save(output_path)

    def _get_model(self):
        base_model = MobileNetV2(include_top=False, 
                                 input_tensor=Input(shape=(self.img_height, self.img_width, 3)),
                                 weights="imagenet")
        custom_model = base_model.output
        #custom_model = GlobalAveragePooling2D()(custom_model)
        custom_model = AveragePooling2D(pool_size=(7,7))(custom_model)
        custom_model = Flatten()(custom_model)
        custom_model = Dense(128, activation="relu")(custom_model)
        custom_model = Dropout(0.5)(custom_model)
        custom_model = Dense(1, activation="sigmoid")(custom_model)

        # freeze layers
        for layer in base_model.layers:
            layer.trainable=False
        
        return Model(inputs=base_model.input, outputs=custom_model)

