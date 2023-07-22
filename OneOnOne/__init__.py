import pickle
import sys

import numpy as np
import pandas as pd
import os, sys, wget
import math
import datetime
import seaborn as sns
import zipfile
from zipfile import ZipFile
import random
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import load_model
# import tensorflow_datasets as tfds
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from pkg_resources import resource_filename
from scipy.stats import randint

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import shuffle

from transformers import pipeline
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import GPT2Tokenizer, GPT2ForQuestionAnswering
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, RobertaForQuestionAnswering
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import AutoTokenizer, ErnieModel
import torch
from torchvision import transforms

from bs4 import BeautifulSoup
import re
from tqdm import keras
import pkgutil
import requests, zipfile
from io import BytesIO
import gdown
import imageio as iio
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

class PretrainedModel:
    def __init__(self, model_type="resnet50", dataset="cifar10", samplingtype="none"):

        self.model_type=model_type.lower()
        self.dataset=dataset.lower()
        self.samplingtype=samplingtype.lower()

        self.map={"resnet50_cifar10_none":"1YVG0lAnpBfM_MB7ctV1vHcslb6O-3Vbm","resnet50_cifar10_leastconfidence":"1fSJYo5VOppTkgWb-Fu2Sf_JL4s9JUmoK","resnet50_cifar10_mixed":"","resnet50_cifar10_margin":"","efficientnetb6_cifar10_none":"16Wqfx7mcEhssZksF1yUAUEG6mkhMSeme","efficientnetb6_tinyimagenet_none":"1pGX8zB99ugqcvPohxykPC1JLM0Ld0L-D"}

        if self.dataset=="tinyimagenet":
            self.model_type="efficientnetb6"
            self.samplingtype="none"

        self.model_name=f'{self.model_type}_{self.dataset}_{self.samplingtype}'

        if not os.path.isdir(os.getcwd()+'/models_to_load/'):
            os.makedirs(os.getcwd()+'/models_to_load/')

        # if self.samplingtype=="none" or self.samplingtype=="None":
        #     self.path=os.getcwd()+f'/models_to_load/{self.model_type}_{self.dataset}'
        # else:
        #     self.path=os.getcwd()+f'/models_to_load/{self.model_type}_{self.dataset}_{self.samplingtype}'

        self.path = os.getcwd() + f'/models_to_load/{self.model_name}'

        try:
            os.system(f"gdown --id {self.map[self.model_name]}")
        except:
            os.system(f"!gdown --id {self.map[self.model_name]}")


        for file in os.listdir(os.getcwd()):
            if file.endswith(f"{self.model_name}.zip"):
                zip_file = ZipFile(file)
                zip_file.extractall(os.getcwd() + f'/models_to_load/')

        self.model=load_model(self.path)
        self.model.summary()


class Classification:
    def __init__(self, model_type="resnet50", batch_size=16, epochs=250, dataset="cifar10", validation_split=0.3, shuffle_bool=True, early_stopping_patience=10, lr_reducer_patience=10):

        self.model_type = model_type.lower()
        self.date=datetime.datetime.now()
        self.dataset=dataset.lower()
        self.shuffle_bool = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_layer_classes=0
        self.input_shape = (32,32,3)
        self.early_stopping_patience=early_stopping_patience
        self.lr_reducer_patience=lr_reducer_patience
        self.validation_split = validation_split
        self.callbacks = self.get_callbacks()
        self.datagen = self.get_datagen()

        if self.dataset=="cifar10":
            self.output_layer_classes = 10
            self.input_shape = (224,224,3)

        elif self.dataset == "cifar100":
            self.output_layer_classes = 100
            self.input_shape = (224, 224, 3)

        elif self.dataset=="tinyimagenet":
            self.output_layer_classes = 200
            self.input_shape = (32, 32, 3)
            self.download_dataset()
            from classes import i2d

        else:
            print("Invalid Input!")

        self.token=input("Train/Load?  :   ")
        self.train_it, self.val_it = self.get_dataset()

        if self.token.lower()=='train':
            print("Training...")

            self.model = self.define_compile_model()
            self.model.summary()

            if self.dataset=="tinyimagenet":
                history = self.model.fit(self.train_it, validation_data=self.val_it,
                                         epochs=self.epochs, verbose=1, workers=4,
                                         callbacks=self.callbacks)
            else:
                history = self.model.fit(self.datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size), validation_data=(self.X_test, self.y_test),
                                         epochs=self.epochs, verbose=1, workers=4,
                                         callbacks=self.callbacks)


            self.model.save(os.getcwd()+f"/trained_models/{self.model_type}_{self.dataset}_{self.date}_completed")

            history_token = input("Save/Discard?  :   ")

            if history_token.lower()=="save":
                with open(f'{self.model_type}_{self.dataset}_{self.date}_history', 'wb') as file_pi:
                    pickle.dump(history.history, file_pi)
                print("Saved.")
            elif history_token.lower()=="save":
                print("Discarded.")
            else:
                print("Invalid Input!")

        elif self.token.lower() == 'load':
            print("Loading...")

            self.model = load_model(os.getcwd()+"/models_to_load/"+f"{self.model_type}_{self.dataset}")

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
        x = tf.keras.layers.Dense(self.output_layer_classes, activation="softmax", name="classification")(x)
        return x

    def feature_extractor(self, inputs):

        if self.model_type == "resnet50":
            feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=self.input_shape,
                                                                      include_top=False,
                                                                      weights='imagenet', classes=self.output_layer_classes,
                                                                      classifier_activation="softmax")(inputs)
        elif self.model_type == "efficientnetb6":
            feature_extractor = tensorflow.keras.applications.EfficientNetB6(input_shape=self.input_shape,
                                                                             include_top=False,
                                                                             weights='imagenet', classes=self.output_layer_classes,
                                                                             classifier_activation="softmax")(inputs)

        elif self.model_type == "densenet":
            feature_extractor = tf.keras.applications.DenseNet201(input_shape=self.input_shape,
                                                                  include_top=False,
                                                                  weights='imagenet', classes=self.output_layer_classes,
                                                                  classifier_activation="softmax")(inputs)
        elif self.model_type == "vgg19":
            feature_extractor = tf.keras.applications.VGG19(input_shape=self.input_shape,
                                                            include_top=False,
                                                            weights='imagenet', classes=self.output_layer_classes,
                                                            classifier_activation="softmax")(inputs)
        else:
            feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=self.input_shape,
                                                                      include_top=False,
                                                                      weights='imagenet',
                                                                      classes=self.output_layer_classes,
                                                                      classifier_activation="softmax")(inputs)
            print("Invalid argument for model type.")

        return feature_extractor

    def final_model(self, inputs):

        if self.dataset=="cifar10":
            resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
            feature_extractor = self.feature_extractor(resize)
            classification_output = self.classifier(feature_extractor)
        elif self.dataset=="tinyimagenet":
            feature_extractor = self.feature_extractor(inputs)
            classification_output = self.classifier(feature_extractor)
        else:
            feature_extractor = self.feature_extractor(inputs)
            classification_output = self.classifier(feature_extractor)

        return classification_output

    def define_compile_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        classification_output = self.final_model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=classification_output)

        model.compile(optimizer='SGD',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def preprocess_image_input(self,input_images):
        if self.model_type=="efficientnetb6":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.efficientnet.preprocess_input(input_images)
        elif self.model_type=="resnet50":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
        elif self.model_type=="densenet":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.densenet.preprocess_input(input_images)
        elif self.model_type=="vgg19":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.vgg19.preprocess_input(input_images)
        else:
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)

        return output_ims

    def get_callbacks(self):

        save_dir = os.getcwd()+f'/trained_models/{self.model_type}_{self.dataset}'
        model_name = self.model_type + '_model_' + str(self.date.day) + '_' + str(self.date.month) + '_' + str(
            self.date.year) + '.{epoch:03d}.h5'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=self.early_stopping_patience,
                                       restore_best_weights=True)

        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=self.lr_reducer_patience,
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
            validation_split=self.validation_split)

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

        if self.dataset=="tinyimagenet":
            train_it = self.datagen.flow_from_directory(os.getcwd()+'/tiny-imagenet-200/train',
                                                        batch_size=self.batch_size, subset="training", shuffle=self.shuffle_bool)
            val_it = self.datagen.flow_from_directory(os.getcwd()+'/tiny-imagenet-200/train', batch_size=self.batch_size,
                                                      subset="validation", shuffle=self.shuffle_bool)
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            # print(number_of_train_samples)
            # print(number_of_val_samples)
        else:
            if self.dataset == "cifar10":
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                num_classes = len(classes)
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

            elif self.dataset == "mnist":
                num_classes = 10
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

            elif self.dataset == "cifar100":
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

            self.X_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it

    def save_history_details(self, history_data, keys):

        pd.DataFrame(history_data).to_csv(os.getcwd()+f"history_{self.model_type}_{self.dataset}_{self.date}.csv", header=keys)

        return

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
        loss, accuracy = self.model.evaluate(self.val_it, batch_size=self.batch_size, verbose=1)


class Sampling:
    def __init__(self, samplingtype, dataset="cifar10", model_type = "resnet50", goal=99, jump=5000, validation_split=0.3, first_data_samples=10000, batch_size = 16, epochs = 250, shuffle_bool = True, early_stopping_patience = 10, lr_reducer_patience = 10):

        self.validation_split=validation_split
        self.model_type = model_type.lower()
        self.epochs=epochs
        self.jump = jump
        self.samplingtype = samplingtype.lower()
        self.goal = goal
        self.shuffle_bool = shuffle
        self.batch_size = batch_size
        self.dataset = dataset.lower()
        self.output_layer_classes = 0
        self.input_shape = (32, 32, 3)
        self.date=datetime.datetime.now()
        self.first_data_samples = first_data_samples
        self.early_stopping_patience = early_stopping_patience
        self.lr_reducer_patience = lr_reducer_patience
        self.callbacks = self.get_callbacks()
        self.datagen = self.get_datagen()
        if self.dataset == "cifar10" or self.dataset == "mnist":
            self.output_layer_classes = 10
            self.input_shape = (224, 224, 3)
        elif self.dataset == "tinyimagenet":
            self.output_layer_classes = 200
            self.input_shape = (32, 32, 3)
            self.download_dataset()
            from classes import i2d
        else:
            print("Invalid Input!")

        self.train_it, self.val_it = self.get_dataset()

        if samplingtype== "random":
          random_num = np.random.randint(self.first_data_samples,self.X_train.shape[0],size=self.X_train.shape[0]-self.first_data_samples)
          self.random_num= random_num

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
        x = tf.keras.layers.Dense(self.output_layer_classes, activation="softmax", name="classification")(x)
        return x

    def feature_extractor(self, inputs):

        if self.model_type == "resnet50":
            feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=self.input_shape,
                                                                      include_top=False,
                                                                      weights='imagenet',
                                                                      classes=self.output_layer_classes,
                                                                      classifier_activation="softmax")(inputs)
        elif self.model_type == "efficientnetb6":
            feature_extractor = tensorflow.keras.applications.EfficientNetB6(input_shape=self.input_shape,
                                                                             include_top=False,
                                                                             weights='imagenet',
                                                                             classes=self.output_layer_classes,
                                                                             classifier_activation="softmax")(inputs)

        elif self.model_type == "densenet":
            feature_extractor = tf.keras.applications.DenseNet201(input_shape=self.input_shape,
                                                                  include_top=False,
                                                                  weights='imagenet', classes=self.output_layer_classes,
                                                                  classifier_activation="softmax")(inputs)
        elif self.model_type == "vgg19":
            feature_extractor = tf.keras.applications.VGG19(input_shape=self.input_shape,
                                                            include_top=False,
                                                            weights='imagenet', classes=self.output_layer_classes,
                                                            classifier_activation="softmax")(inputs)
        else:
            feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=self.input_shape,
                                                                      include_top=False,
                                                                      weights='imagenet',
                                                                      classes=self.output_layer_classes,
                                                                      classifier_activation="softmax")(inputs)
            print("Invalid argument for model type.")

        return feature_extractor

    def final_model(self, inputs):

        if self.dataset == "cifar10":
            resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
            feature_extractor = self.feature_extractor(resize)
            classification_output = self.classifier(feature_extractor)
        elif self.dataset == "tinyimagenet":
            feature_extractor = self.feature_extractor(inputs)
            classification_output = self.classifier(feature_extractor)
        else:
            feature_extractor = self.feature_extractor(inputs)
            classification_output = self.classifier(feature_extractor)

        return classification_output

    def define_compile_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        classification_output = self.final_model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=classification_output)

        model.compile(optimizer='SGD',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def preprocess_image_input(self, input_images):
        if self.model_type == "efficientnetb6":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.efficientnet.preprocess_input(input_images)
        elif self.model_type == "resnet50":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
        elif self.model_type == "densenet":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.densenet.preprocess_input(input_images)
        elif self.model_type == "vgg19":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.vgg19.preprocess_input(input_images)
        else:
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)

        return output_ims

    def get_callbacks(self):

        save_dir = os.getcwd()+f'/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}'
        model_name = self.model_type + '_model_' + str(self.date.day) + '_' + str(self.date.month) + '_' + str(
            self.date.year) + '.{epoch:03d}.h5'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                                       patience=self.early_stopping_patience,
                                       restore_best_weights=True)

        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=self.lr_reducer_patience,
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
            validation_split=self.validation_split)

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

        if self.dataset == "tinyimagenet":
            train_it = self.datagen.flow_from_directory(os.getcwd()+'/tiny-imagenet-200/train',
                                                        batch_size=self.batch_size, subset="training",
                                                        shuffle=self.shuffle_bool)
            val_it = self.datagen.flow_from_directory(os.getcwd()+'/tiny-imagenet-200/train',
                                                      batch_size=self.batch_size,
                                                      subset="validation", shuffle=self.shuffle_bool)
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            # print(number_of_train_samples)
            # print(number_of_val_samples)

        else:

            shuffle_random_token = input("Do you want to shuffle the training data? (yes/no):  ")

            if shuffle_random_token.lower() == "yes":
                seed_token = input("Enter random seed (int):  ")
            else:
                seed_token = float('inf')

            if self.dataset == "cifar10":
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                num_classes = len(classes)
                (training_images, training_labels), (
                validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if seed_token!=float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.dataset == "mnist":
                num_classes=10
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if seed_token != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.dataset == "cifar100":
                num_classes=100
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.cifar100.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if seed_token != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            training_labels = to_categorical(training_labels, num_classes)
            validation_labels = to_categorical(validation_labels, num_classes)
            self.datagen.fit(train_X)

            train_it=self.datagen.flow(train_X, training_labels, batch_size=self.batch_size)
            val_it = (valid_X, validation_labels)

            self.X_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it


    def get_iterations_rs(self):

        e=[0,0]

        k=0

        num=0

        y=[]
        x=[]
        for i in range(0,self.first_data_samples):
          y.append(self.y_train[i])
          x.append(self.X_train[i])

        while (e[1] < self.goal / 100)&(k + self.jump < self.X_train.shape[0]):

            for i in range(0+k,self.jump+k):
                x.append(self.X_train[self.random_num[i]])
                y.append(self.y_train[self.random_num[i]])

            k=k+self.jump
            num=num+1

            self.model.fit(np.array(x), np.array(y), epochs=50)

            e = self.model.evaluate(self.X_test, self.y_test)

        return e[1],num


    def get_entropy(self,y_predicted_en):
        sum_prob = 0
        entropy = []

        for i in range(0,y_predicted_en.shape[0]):
            for j in range(0,y_predicted_en.shape[1]):
              if(y_predicted_en[i][j]==0):
                continue
              k = y_predicted_en[i][j]*math.log(y_predicted_en[i][j])
              sum_prob = sum_prob+k
            entropy.append([int(i),-(sum_prob)])
            sum_prob=0
            k=0

        entropy.sort(key=lambda x:x[1],reverse=True)

        return entropy

    def get_lc(self,y_predicted_lc):
        probarray = []

        for i in range(0,y_predicted_lc.shape[0]):
            probarray.append([int(i),np.max(y_predicted_lc[i])])

        probarray.sort(key=lambda x:x[1])

        return probarray

    def get_ratio(self,y_predicted_r):
        ratio_array = []

        for i in range(0,y_predicted_r.shape[0]):
            y=y_predicted_r[i]
            sorted(y,reverse=True)
            first_max=y[0]
            second_max=y[1]
            if second_max==0:
              continue
            ratio_array.append([int(i),first_max/second_max])

        ratio_array.sort(key=lambda x:x[1])

        return ratio_array


    def get_margin(self,y_predicted_margin):
        a=[]
        b=[]
        for i in range(0,y_predicted_margin.shape[0]):
            a=y_predicted_margin[i]
            c=sorted(a,reverse=True)
            difference=c[0]-c[1]

            b.append([int(i),difference])

        b.sort(key=lambda x:x[1])

        return b

    def get_hc(self,y_predicted_hc):
        probarray = []

        for i in range(0,y_predicted_hc.shape[0]):
            probarray.append([int(i),np.max(y_predicted_hc[i])])

        probarray.sort(key=lambda x:x[1],reverse=True)

        return probarray

    # def load_trained_model(self):

    def initial_training(self):

        print("Training...")

        if self.dataset=="tinyimagenet":
            self.model = self.define_compile_model()
            self.model.summary()

            history = self.model.fit(self.train_it, validation_data=self.val_it,
                                     epochs=self.epochs, verbose=1, workers=4,
                                     callbacks=self.callbacks)

            self.model.save(os.getcwd() + f"/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}_{self.first_data_samples}_{self.date}_initial_training")


            with open(f'{self.model_type}_{self.dataset}_{self.date}_history', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            print("Saved.")

        else:

            self.model = self.define_compile_model()
            self.model.summary()

            history = self.model.fit(self.datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size), validation_data=(self.X_test, self.y_test),
                                     epochs=self.epochs, verbose=1, workers=4,
                                     callbacks=self.callbacks)
            self.model.save(os.getcwd() + f"/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}_{self.first_data_samples}_{self.date}_initial_training")

            with open(f'{self.model_type}_{self.dataset}_{self.date}_history', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            print("saved.")



    def get_iterations(self):

        eval_metrics=[0,0]

        num=0
        batch_size_data = [0]
        acuracy_data = [0]
        epoch_data = []
        history_data = []

        y = []
        x = []

        if self.samplingtype== "random":
          eval_metrics[1],num = self.get_iterations_rs()

          return eval_metrics[1],num

        X_train_copy=self.X_train[self.first_data_samples:self.X_train.shape[0]]
        y_train_copy=self.y_train[self.first_data_samples:self.X_train.shape[0]]     

        for i in range(0,self.first_data_samples):
          y.append(self.y_train[i])
          x.append(self.X_train[i])

        while (eval_metrics[1] < self.goal / 100)&(self.jump <= X_train_copy.shape[0]):

            total_index=[*range(0, X_train_copy.shape[0], 1)]
            y_predicted = self.model.predict(X_train_copy)

            if self.samplingtype== "margin":
              values = self.get_margin(y_predicted)
            elif self.samplingtype== "leastconfidence":
              values = self.get_lc(y_predicted)
            elif self.samplingtype== "highestconfidence":
              values = self.get_hc(y_predicted)
            elif self.samplingtype== "entropy":
              values = self.get_entropy(y_predicted)
            elif self.samplingtype== "ratio":
              values = self.get_ratio(y_predicted)
            elif self.samplingtype== "mixed":
              values_lc = self.get_lc(y_predicted)
              values_hc = self.get_hc(y_predicted)
            else:
              print("Invalid Input!")

            index = []

            if self.samplingtype== "mixed":
              for i in range(0, self.jump // 2):
                index.append(values_lc[i][0])
                x.append(X_train_copy[values_lc[i][0]])
                y.append(y_train_copy[values_lc[i][0]])

              for i in range(0, self.jump // 2):
                index.append(values_hc[i][0])
                x.append(X_train_copy[values_hc[i][0]])
                y.append(y_train_copy[values_hc[i][0]])


            else:
              for i in range(0,self.jump):
                index.append(values[i][0])
                x.append(X_train_copy[values[i][0]])
                y.append(y_train_copy[values[i][0]])

            num+=1

            h=self.model.fit(self.datagen.flow(np.array(x), np.array(y), batch_size=self.batch_size), validation_data=self.val_it,
                                     epochs=self.epochs, verbose=1, workers=4,
                                     callbacks=self.callbacks)

            batch_size_data.append(np.array(x).shape[0])
            epoch_data.append(num)
            history_data.append(h)

            eval_metrics = self.model.evaluate(self.X_test, self.y_test)
            print(eval_metrics)
            acuracy_data.append(100*float(eval_metrics[1]))

            a=[]
            for element in total_index:
              if element not in index:
                a.append(element)

            X_train_copy=X_train_copy[a]
            y_train_copy=y_train_copy[a]

        self.model.save(os.getcwd() + f"/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}_{self.date}_completed")

        return history_data,epoch_data,batch_size_data,acuracy_data

class HTMLparser:
    def __init__(self, words):
        self.words = words
    def clean_html(self,raw_html):
        clean_brackets = re.compile('<.*?>')
        cleantext = re.sub(clean_brackets, '', raw_html)
        cleantext = re.sub(' ,', '', cleantext)
        cleantext = re.sub('\n', '', cleantext)
        cleantext = cleantext.replace('\\', '')

        return cleantext

    def produce_text(self,word):
        link = "https://en.wikipedia.org/wiki/" + word

        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')

        p=soup.find_all('p')

        context=str(p)

        cleaned_text=self.clean_html(context)

        return cleaned_text


    def get_context_text(self):

        full_context = ""

        for word in self.words:

            text_for_one_word = self.produce_text(word)
            full_context = full_context + text_for_one_word[3:-1]

        return full_context


class ContextDecider:
    def __init__(self, load=False, user_input=False, dataset="tinyimagenet", model_type="efficientnetb6", samplingtype="none", threshold=0.2, validation_split=0.3, batch_size=16, shuffle_bool=True):

        self.load=load
        self.user_input=user_input
        self.dataset = dataset.lower()
        self.model_type = model_type.lower()
        self.samplingtype = samplingtype
        self.threshold = threshold
        self.validation_split=validation_split
        self.batch_size = batch_size
        self.shuffle_bool = shuffle_bool
        self.datagen = self.get_datagen()

        if self.dataset == "cifar10" or self.dataset == "mnist":
            self.output_layer_classes = 10
            self.input_shape = (224, 224, 3)

        elif self.dataset == "cifar100":
            self.output_layer_classes = 100
            self.input_shape = (224, 224, 3)

        elif self.dataset == "tinyimagenet":
            self.output_layer_classes = 200
            self.input_shape = (32, 32, 3)
            self.download_dataset()
            from classes import i2d

        else:
            print("Invalid Input!")

        self.train_it, self.val_it = self.get_dataset()

        if self.load:
            path=input("Please input the model's path/name in the current directory (str):     ")
            self.contextmodel=load_model(os.getcwd()+path)
        else:
            self.contextmodel = PretrainedModel(model_type=self.model_type, dataset=self.dataset,
                                          samplingtype=self.samplingtype)
        # self.pretrained.model
        if self.user_input:
            path=input("Please enter image path in the current directory (str):    ")
            img = Image.open(os.getcwd()+f"{path}")
            img.show()
            img = iio.imread(os.getcwd()+f"{path}")
            processed_image = preprocess_image_input(img)
            self.pred = self.contextmodel.model.predict(processed_image)
        else:
            self.pred = self.contextmodel.model.predict_generator(self.val_it, random.randint(1, 3000))

    def preprocess_image_input(self,input_images):
        if self.model_type=="efficientnetb6":
            input_images = input_images.astype('float32')
            output_ims = tf.keras.applications.efficientnet.preprocess_input(input_images)

        return output_ims

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
            validation_split=self.validation_split)

        return datagen

    def decide_context(self):

        from classes import i2d

        predicted_list = []
        classes_prob_list = []
        prediction_index = []
        final_classes = []

        output = []

        labels = self.val_it.class_indices
        labels2 = dict((v, k) for k, v in labels.items())

        print(f"{self.pred.shape[0]}")
        for i in range(0, self.pred.shape[0]):
            classes_prob_list = []
            for j in range(0, self.output_layer_classes):
                classes_prob_list.append([int(j), self.pred[i][j]])

            classes_prob_list.sort(key=lambda x: x[1], reverse=True)

            first_highest_predicted_classes = classes_prob_list[0][0]
            first_highest_predicted_class_confidence = classes_prob_list[0][1]

            second_highest_predicted_classes = classes_prob_list[1][0]
            second_highest_predicted_class_confidence = classes_prob_list[1][1]

            predicted_list.append([first_highest_predicted_classes, first_highest_predicted_class_confidence,
                                   second_highest_predicted_classes, second_highest_predicted_class_confidence])

        if (predicted_list[0][1] - predicted_list[0][3]) >= self.threshold:
            prediction_index.append(predicted_list[0][0])
        elif (predicted_list[0][1] - predicted_list[0][3]) < self.threshold:
            prediction_index.append(predicted_list[0][0])
            prediction_index.append(predicted_list[0][2])

        for i in range(0, len(prediction_index)):
            final_classes.append(i2d[labels2[prediction_index[i]]])

        for ele in final_classes:
            b = ele.split(',')
            output = output + b

        return output

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

        if self.dataset == "tinyimagenet":
            train_it = self.datagen.flow_from_directory(os.getcwd() + '/tiny-imagenet-200/train',
                                                        batch_size=self.batch_size, subset="training",
                                                        shuffle=self.shuffle_bool)
            val_it = self.datagen.flow_from_directory(os.getcwd() + '/tiny-imagenet-200/train',
                                                      batch_size=self.batch_size,
                                                      subset="validation", shuffle=self.shuffle_bool)
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            # print(number_of_train_samples)
            # print(number_of_val_samples)
        else:
            if self.dataset == "cifar10":
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                num_classes = len(classes)
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

            elif self.dataset == "mnist":
                num_classes = 10
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

            elif self.dataset == "cifar100":
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

            self.X_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it


class QuestionAnswer:
    def __init__(self, context, chatbot="bert"):

        self.exit_commands = ("no", "n", "quit", "pause", "exit", "goodbye", "bye", "later", "stop")
        self.positive_commands = ("y", "yes", "yeah", "sure", "yup", "ya", "probably", "maybe")
        self.context = context
        self.max_length = len(self.context)
        self.chatbot = chatbot.lower()

        if self.chatbot == "bert":
            self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        elif self.chatbot == "gpt2":
            self.model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        elif self.chatbot == "roberta":
            self.model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
            self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

        # elif self.chatbot == "ernie":
        #     self.model = ErnieModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
        #     self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")

        # elif self.chatbot == "vqa":
        #     self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
        #     self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

        else:
            print("invalid input")

    def question_answer(self, question):

        if self.chatbot=="bert":
            c = self.context[:512]

            input_ids = self.tokenizer.encode(question, c, add_special_tokens=True, truncation=True, max_length=len(context))

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # tokens = self.tokenizer.tokenize(self.context, max_length=self.max_length, truncation=True)

            sep_idx = input_ids.index(self.tokenizer.sep_token_id)

            num_seg_a = sep_idx + 1

            num_seg_b = len(input_ids) - num_seg_a

            segment_ids = [0] * num_seg_a + [1] * num_seg_b

            assert len(segment_ids) == len(input_ids)

            output = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

            answer_start = torch.argmax(output.start_logits)
            answer_end = torch.argmax(output.end_logits)

            if answer_end >= answer_start:
                answer = tokens[answer_start]
                for i in range(answer_start + 1, answer_end + 1):
                    if tokens[i][0:2] == "##":
                        answer += tokens[i][2:]
                    else:
                        answer += " " + tokens[i]

            if answer.startswith("[CLS]") or answer=="":
                answer = "Sorry! Unable to find the answer to your question. Please ask another question."

            answer = f"\nAnswer:\n{format(answer.capitalize())}"

        elif self.chatbot=="gpt2" or self.chatbot=="roberta":
            inputs = self.tokenizer(question, self.context, return_tensors="pt", truncation='longest_first', )

            input_ids = self.tokenizer.encode(question, self.context, max_length=self.max_length)

            with torch.no_grad():
                outputs = self.model(**inputs)

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            output_ids = self.tokenizer.decode(predict_answer_tokens)

            if output_ids == "" or output_ids.startswith("[CLS]") or output_ids== "<s>":
                output_ids = "Sorry! Unable to find the answer to your question. Please ask another question."

            answer = f"\nAnswer:\n {format(output_ids.capitalize())}"

        return answer

    def ask(self):

        while True:
            flag = True
            question = input("\nPlease enter your question: \n")

            if question.lower() in self.exit_commands:
                print("\nBye!")
                flag = False

            if not flag:
                break

            print(self.question_answer(question))





