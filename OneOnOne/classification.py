import random
import pandas as pd
import os, sys, wget
import time
import datetime
import numpy as np
import math
import datetime
from tensorflow.keras.models import load_model

from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import shuffle

import tensorflow as tf
import tensorflow.keras
import torch

import warnings

import copy
from bayes_opt import BayesianOptimization
tf.config.run_functions_eagerly(True)

class Classification:
    def __init__(self, file_input_params=False, filepath="", train_or_load="train", model_type="resnet50", batch_size=16, epochs=250, dataset="cifar10", validation_split=0.3, shuffle_bool=True, early_stopping_patience=10, lr_reducer_patience=10, sampling_type="none"):
        warnings.filterwarnings("ignore")

        self.model_params = {}

        numeric_args = ["goal", "jump", "first_data_samples", "batch_size", "epochs", "early_stopping_patience", "lr_reducer_patience"]
        boolean_args = ["shuffle_bool", "save_and_reload_bool", "load_saved_model_bool"]
        float_args = ["validation_split"]

        self.model_params["validation_split"] = validation_split
        self.model_params["model_type"] = model_type.lower()
        self.model_params["epochs"] = epochs
        self.model_params["jump"] = jump
        self.model_params["sampling_type"] = sampling_type.lower()
        self.model_params["goal"] = goal
        self.model_params["shuffle_bool"] = shuffle_bool
        self.model_params["batch_size"] = batch_size
        self.model_params["dataset"] = dataset
        self.model_params["load_saved_model_bool"] = load_saved_model_bool
        self.model_params["save_and_reload_bool"] = save_and_reload_bool
        self.model_params["first_data_samples"] = first_data_samples
        self.model_params["early_stopping_patience"] = early_stopping_patience
        self.model_params["lr_reducer_patience"] = lr_reducer_patience
        self.model_params["shuffle_training_data_seed"] = shuffle_training_data_seed
        self.model_params["shuffle_training_data_bool"] = shuffle_training_data_bool

        if file_input_params:

            with open(filepath) as f:
                lines = [line.rstrip() for line in f]

            for line in lines:
                var, val = line.split(" ")
                if var in numeric_args:
                    val = int(val)
                elif var in float_args:
                    val = float(val)
                elif var in boolean_args:
                    if val == "True":
                        val = True
                    else:
                        val = False
                self.model_params[var] = val


        self.date = datetime.datetime.now()

        if self.model_params["dataset"] =="cifar10" or self.model_params["dataset"] =="mnist":
            output_layer_classes = 10
            input_shape = (224,224,3)

        elif self.model_params["dataset"] == "cifar100":
            output_layer_classes = 100
            input_shape = (224, 224, 3)

        elif self.model_params["dataset"] =="tinyimagenet":
            output_layer_classes = 200
            input_shape = (32, 32, 3)
            self.download_dataset()
            from classes import i2d

        else:
            print("Invalid Input!")
        # self.input_shape = (32,32,3)

        self.model_params["output_layer_classes"] = output_layer_classes
        self.model_params["input_shape"] = input_shape

        self.callbacks = self.get_callbacks()
        self.datagen = self.get_datagen()

        self.training_dir = os.getcwd() + f"/trained_models/"+str(self.model_params["model_type"])+"/"+str(self.model_params["dataset"])+"/"+self.model_params["sampling_type"]+"_"+str(self.model_params["first_data_samples"])+"k"

        if not os.path.isdir(self.training_dir):
            os.makedirs(self.training_dir)

        self.train_it, self.val_it = self.get_dataset()

        if self.model_params["train_or_load"].lower()=='train':
            print("Training...")

            self.model = self.define_compile_model()
            self.model.summary()

            if self.model_params["dataset"] == "tinyimagenet":
                history = self.model.fit(self.train_it, validation_data=self.val_it,
                                         epochs=self.model_params["epochs"], verbose=1, workers=4,
                                         callbacks=self.callbacks)
            else:
                history = self.model.fit(self.datagen.flow(self.x_train, self.y_train, batch_size=self.model_params["batch_size"]), validation_data=(self.X_test, self.y_test),
                                         epochs=self.model_params["epochs"], verbose=1, workers=4,
                                         callbacks=self.callbacks)

            self.model.save(self.training_dir)

            with open(f'{self.model_params["model_type"]}_{self.model_params["dataset"]}_{self.date}_history', 'wb') as handle:
                pickle.dump(history, handle)
            print("Saved.")

        elif self.model_params["train_or_load"].lower() == 'load':
            print("Loading...")

            self.model = load_model(self.training_dir)

            PRINT("Loading Completed.")

        else:
            print("Invalid Input!")

    def lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning Rate: ', lr)

        return lr

    def classifier(self, inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.model_params["output_layer_classes"], activation="softmax", name="classification")(x)
        return x

    def feature_extractor(self, inputs):

        if self.model_params["model_type"] == "resnet50":
            feature_extractor = tf.keras.applications.resnet50.ResNet50(input_shape=self.model_params["input_shape"],
                                                                      include_top=False,
                                                                      weights='imagenet', classes=self.model_params["output_layer_classes"],
                                                                      classifier_activation="softmax")(inputs)
        elif self.model_params["model_type"] == "efficientnetb6":
            feature_extractor = tensorflow.keras.applications.EfficientNetB6(input_shape=self.model_params["input_shape"],
                                                                             include_top=False,
                                                                             weights='imagenet', classes=self.model_params["output_layer_classes"],
                                                                             classifier_activation="softmax")(inputs)

        elif self.model_params["model_type"] == "densenet":
            feature_extractor = tf.keras.applications.DenseNet201(input_shape=self.model_params["input_shape"],
                                                                  include_top=False,
                                                                  weights='imagenet', classes=self.model_params["output_layer_classes"],
                                                                  classifier_activation="softmax")(inputs)
        elif self.model_params["model_type"] == "vgg19":
            feature_extractor = tf.keras.applications.VGG19(input_shape=self.model_params["input_shape"],
                                                            include_top=False,
                                                            weights='imagenet', classes=self.model_params["output_layer_classes"],
                                                            classifier_activation="softmax")(inputs)
        else:
            feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=self.model_params["input_shape"],
                                                                      include_top=False,
                                                                      weights='imagenet',
                                                                      classes=self.model_params["output_layer_classes"],
                                                                      classifier_activation="softmax")(inputs)
            print("Invalid argument for model type.")

        return feature_extractor

    def final_model(self, inputs):

        if self.model_params["dataset"] == "cifar10":
            resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
            feature_extractor = self.feature_extractor(resize)
            classification_output = self.classifier(feature_extractor)
        elif self.model_params["dataset"] == "tinyimagenet":
            feature_extractor = self.feature_extractor(inputs)
            classification_output = self.classifier(feature_extractor)
        else:
            feature_extractor = self.feature_extractor(inputs)
            classification_output = self.classifier(feature_extractor)

        return classification_output

    def define_compile_model(self):
        inputs = tf.keras.layers.Input(shape=(32,32,3))

        classification_output = self.final_model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=classification_output)

        model.compile(optimizer='SGD',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def preprocess_image_input(self,input_images):
        if self.model_params["model_type"] == "efficientnetb6":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.efficientnet.preprocess_input(input_images)
        elif self.model_params["model_type"] == "resnet50":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
        elif self.model_params["model_type"] == "densenet":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.densenet.preprocess_input(input_images)
        elif self.model_params["model_type"] == "vgg19":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.vgg19.preprocess_input(input_images)
        else:
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)

        return output_ims

    def get_callbacks(self):

        callback_dir = os.getcwd() + f'/saved_callbacks/{self.model_params["model_type"]}/{self.model_params["dataset"]}/{self.model_params["sampling_type"]}_{self.model_params["first_data_samples"]}k'

        # save_dir = os.getcwd()+f'/trained_models/{self.model_type}/{self.dataset}/{"no sampling"}'
        model_name = self.model_params["model_type"] + '_model_' + str(self.date.day) + '_' + str(self.date.month) + '_' + str(
            self.date.year) + '.{epoch:03d}.h5'

        if not os.path.isdir(callback_dir):
            os.makedirs(callback_dir)
        filepath = os.path.join(callback_dir, model_name)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)

        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=self.model_params["early_stopping_patience"],
                                       restore_best_weights=True)

        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=self.model_params["lr_reducer_patience"],
                                       min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping]

        return callbacks

    def get_datagen(self):

        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=self.model_params["validation_split"])

        return datagen

    def download_dataset(self):

        print("Extracting...")
        if not 'tiny-imagenet-200.zip' in os.listdir(os.getcwd()):
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            tiny_imgdataset = wget.download(url, out=os.getcwd())

        for file in os.listdir(os.getcwd()):
            if file.endswith("tiny-imagenet-200.zip"):
                zip_file = ZipFile(file)
                zip_file.extractall()

        try:
            os.system("gdown --id 1JgRlpet7-P-x7Exweb8HC-zUcYsF5fGN")
        except:
            os.system("!gdown --id 1JgRlpet7-P-x7Exweb8HC-zUcYsF5fGN")

        print("Done.")

    def get_dataset(self):

        if self.model_params["dataset"] == "tinyimagenet":
            train_it = self.datagen.flow_from_directory(os.getcwd()+'/tiny-imagenet-200/train',
                                                        batch_size=self.model_params["batch_size"], subset="training", shuffle=self.model_params["shuffle_bool"])
            val_it = self.datagen.flow_from_directory(os.getcwd()+'/tiny-imagenet-200/train', batch_size=self.model_params["batch_size"],
                                                      subset="validation", shuffle=self.model_params["shuffle_bool"])
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            # print(number_of_train_samples)
            # print(number_of_val_samples)
        else:
            if self.model_params["dataset"] == "cifar10":
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                num_classes = len(classes)
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

            elif self.model_params["dataset"] == "mnist":
                num_classes = 10
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

            elif self.model_params["dataset"] == "cifar100":
                num_classes = 100
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.cifar100.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

            training_labels = to_categorical(training_labels, num_classes)
            validation_labels = to_categorical(validation_labels, num_classes)
            self.datagen.fit(train_X)

            train_it = self.datagen.flow(train_X, training_labels, batch_size=self.batch_size)
            val_it = (valid_X, validation_labels)

            self.x_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it

    # def save_history_details(self, history_data, keys):
    #
    #     pd.DataFrame(history_data).to_csv(os.getcwd()+f"history_{self.model_type}_{self.dataset}_{self.date}.csv", header=keys)
    #
    #     return

    def get_history_details(self,history):
        keys = []
        total_list_parameter_history = []

        for i in range(0, len(history)):
            individual_keys_list_history = []
            for key in history[i].history.keys():
                if i == 0:
                    keys.append(key)
                a = history[i].history[key]
                individual_keys_list_history.append(a)

            total_list_parameter_history.append(individual_keys_list_history)

        return keys, total_list_parameter_history

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.val_it, batch_size=self.model_params["batch_size"], verbose=1)

