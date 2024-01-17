import pickle
import sys

import numpy as np
import pandas as pd
import os, sys, wget
import math
import time
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

# from transformers import pipeline
# from transformers import BlipProcessor, BlipForQuestionAnswering
# from transformers import GPT2Tokenizer, GPT2ForQuestionAnswering
# from transformers import AutoTokenizer, AutoModel
# from transformers import AutoTokenizer, RobertaForQuestionAnswering
# from transformers import BertForQuestionAnswering, BertTokenizer
# from transformers import AutoTokenizer, ErnieModel
# from transformers import pipeline, Conversation

import torch
# from torchvision import transforms
#
# from bs4 import BeautifulSoup
# import re
# from tqdm import keras
# import pkgutil
# import requests, zipfile
# from io import BytesIO
# import gdown

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import warnings

import copy
# from bayes_opt import BayesianOptimization
tf.config.run_functions_eagerly(True)

# from pydub import AudioSegment
# import speech_recognition as sr
# import pyttsx3
# import pytesseract
# from googletrans import Translator
# from io import BytesIO
# from base64 import b64decode
# from google.colab import output
# from IPython.display import Javascript
# import PyPDF2

class PretrainedModel:
    def __init__(self, model_type="resnet50", dataset="cifar10", sampling_type="none", first_data_samples=0):
        warnings.filterwarnings("ignore")

        self.model_type=model_type.lower()
        self.dataset=dataset.lower()
        self.sampling_type=sampling_type.lower()
        self.first_data_samples=first_data_samples

        self.map={"resnet50_cifar10_none_0k":"1YVG0lAnpBfM_MB7ctV1vHcslb6O-3Vbm","resnet50_cifar10_leastconfidence_0k":"1fSJYo5VOppTkgWb-Fu2Sf_JL4s9JUmoK","resnet50_cifar10_none_10k":"1NOicX1WgVzgRPWwWM_LzPyNvnu5st5Kd","resnet50_cifar10_mixedbayes_0k":"","resnet50_cifar10_margin_0k":"","efficientnetb6_cifar10_none_0k":"16Wqfx7mcEhssZksF1yUAUEG6mkhMSeme","efficientnetb6_tinyimagenet_none_0k":"1pGX8zB99ugqcvPohxykPC1JLM0Ld0L-D"}

        if self.dataset=="tinyimagenet":
            self.model_type="efficientnetb6"
            self.sampling_type="none"

        self.training_dir = os.getcwd() + f"/trained_models/{self.model_type}/{self.dataset}/{self.sampling_type}_{first_data_samples}k"

        self.model_name=f'{self.model_type}_{self.dataset}_{self.sampling_type}_initial_{first_data_samples}k'

        if not os.path.isdir(os.getcwd()+'/trained_models/'):
            os.makedirs(os.getcwd()+'/trained_models/')

        self.path = os.getcwd() + f'/trained_models/{self.model_name}'

        try:
            os.system(f"gdown --id {self.map[self.model_name]}")
        except:
            os.system(f"!gdown --id {self.map[self.model_name]}")


        for file in os.listdir(self.training_dir):
            if file.endswith(f"{self.model_name}.zip"):
                zip_file = ZipFile(file)
                zip_file.extractall(self.training_dir)

    def get_model(self):

        self.model=load_model(self.path)
        self.model.summary()

        return self.model


class Classification:
    def __init__(self, file_input_params=False, filepath="", train_or_load="train", model_type="resnet50", batch_size=16, epochs=250, dataset="cifar10", validation_split=0.3, shuffle_bool=True, early_stopping_patience=10, lr_reducer_patience=10, sampling_type="none"):
        warnings.filterwarnings("ignore")

        self.model_params = {}

        numeric_args = ["goal", "jump", "validation_split", "first_data_samples", "batch_size", "epochs",
                        "early_stopping_patience", "lr_reducer_patience"]
        boolean_args = ["shuffle_bool", "save_and_reload_bool", "load_saved_model_bool"]

        if file_input_params:

            with open(filepath) as f:
                lines = [line.rstrip() for line in f]

            for line in lines:
                var, val = line.split(" ")
                if var in numeric_args:
                    val = int(val)
                elif var in boolean_args:
                    if val == "True":
                        val = True
                    else:
                        val = False
                self.model_params[var] = val

        else:
            self.model_params["validation_split"] = validation_split
            self.model_params["model_type"] = model_type.lower()
            self.model_params["epochs"] = epochs
            self.model_params["jump"] = jump
            self.model_params["shuffle_bool"] = shuffle_bool
            self.model_params["batch_size"] = batch_size
            self.model_params["dataset"] = dataset.lower()
            self.model_params["load_saved_model"] = load_saved_model_bool
            self.model_params["save_and_reload"] = save_and_reload_bool
            self.model_params["first_data_samples"] = first_data_samples
            self.model_params["early_stopping_patience"] = early_stopping_patience
            self.model_params["lr_reducer_patience"] = lr_reducer_patience
            self.model_params["train_or_load"] = train_or_load
            self.model_params["sampling_type"] = sampling_type

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

        callback_dir = os.getcwd() + f'/saved_callbacks/{self.model_params["model_type"]}/{self.model_params["dataset"]}/{self.model_params["sampling_type"]}_{self.model_params["first_data_samples"]}'

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


class Sampling:
    def __init__(self, file_input_params=False, filepath="", sampling_type="mixedbayes", dataset="cifar10", model_type = "resnet50", goal=99, jump=5000, validation_split=0.3, first_data_samples=10000, batch_size = 16, epochs = 250, shuffle_bool = True, early_stopping_patience = 10, lr_reducer_patience = 10, save_and_reload_bool=True,load_saved_model_bool=False, shuffle_training_data_bool=False, shuffle_training_data_seed=float('inf')):
        warnings.filterwarnings("ignore")

        self.model_params={}

        numeric_args=["goal","jump","validation_split","first_data_samples", "batch_size", "epochs", "early_stopping_patience", "lr_reducer_patience", "shuffle_training_data_seed"]
        boolean_args=["shuffle_bool", "save_and_reload_bool", "load_saved_model_bool", "shuffle_training_data_bool"]

        if file_input_params:

            with open(filepath) as f:
                lines = [line.rstrip() for line in f]

            for line in lines:
                var, val = line.split(" ")
                if var in numeric_args:
                    val=int(val)
                elif var in boolean_args:
                    if val=="True":
                        val=True
                    else:
                        val=False
                self.model_params[var]=val

        else:
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

        if self.model_params["dataset"] == "cifar10" or self.model_params["dataset"] == "mnist":
            output_layer_classes = 10
            input_shape = (224, 224, 3)
        elif self.model_params["dataset"] == "cifar100":
            output_layer_classes = 100
            input_shape = (224, 224, 3)
        elif self.model_params["dataset"] == "tinyimagenet":
            output_layer_classes = 200
            input_shape = (32, 32, 3)
            self.download_dataset()
            from classes import i2d
        else:
            print("Invalid Input!")

        self.model_params["output_layer_classes"] = output_layer_classes
        self.model_params["input_shape"] = input_shape
        self.date = datetime.datetime.now()


        self.training_dir = os.getcwd() + f"/trained_models/"+str(self.model_params["model_type"])+"/"+str(self.model_params["dataset"])+"/"+str(self.model_params["sampling_type"])+"_"+str(self.model_params["first_data_samples"])+"k"

        if not os.path.isdir(self.training_dir):
            os.makedirs(self.training_dir)

        with open(self.training_dir + '/model_params.pickle', 'wb') as handle:
            pickle.dump(self.model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.callbacks = self.get_callbacks()
        self.datagen = self.get_datagen()

        self.train_it, self.val_it = self.get_dataset()

        if self.model_params["save_and_reload_bool"]:
            self.save_and_reload_dir = os.getcwd() + f'/reload_params/{self.model_params["model_type"]}/{self.model_params["dataset"]}/{self.model_params["sampling_type"]}_{self.model_params["first_data_samples"]}'

            if not os.path.isdir(self.save_and_reload_dir):
                os.makedirs(self.save_and_reload_dir)

        if not self.model_params["load_saved_model_bool"]:
            self.model_sampling_data = {"accuracy":[[0,0]],"timestamps":[],"history":[],"x_train":[],"y_train":[],"x":[],"y":[]}
            if self.model_params["sampling_type"] == "mixedbayes":
                self.model_sampling_data["least_confidence_jump"] = []
                self.model_sampling_data["high_confidence_jump"] = []
        else:
            with open('model_sampling_data.pickle', 'rb') as handle:
                self.model_sampling_data = pickle.load(handle)

            with open('model_params.pickle', 'rb') as handle:
                self.model_params = pickle.load(handle)

            # self.model=load_model()

        if self.model_params["sampling_type"] == "random":
          random_num = np.random.randint(self.model_params["first_data_samples"],self.x_train.shape[0],size=self.x_train.shape[0]-self.model_params["first_data_samples"])
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
        x = tf.keras.layers.Dense(self.model_params["output_layer_classes"], activation="softmax", name="classification")(x)
        return x

    def feature_extractor(self, inputs):

        if self.model_params["model_type"] == "resnet50":
            feature_extractor = tf.keras.applications.resnet50.ResNet50(input_shape=self.model_params["input_shape"],
                                                                      include_top=False,
                                                                      weights='imagenet',
                                                                      classes=self.model_params["output_layer_classes"],
                                                                      classifier_activation="softmax")(inputs)
        elif self.model_params["model_type"] == "efficientnetb6":
            feature_extractor = tensorflow.keras.applications.EfficientNetB6(input_shape=self.model_params["input_shape"],
                                                                             include_top=False,
                                                                             weights='imagenet',
                                                                             classes=self.model_params["output_layer_classes"],
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
            resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)
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

    def preprocess_image_input(self, input_images):
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

        callback_dir = os.getcwd() + f'/saved_callbacks/{self.model_params["model_type"]}/{self.model_params["dataset"]}/{self.model_params["sampling_type"]}_{self.model_params["first_data_samples"]}'

        if not os.path.isdir(callback_dir):
            os.makedirs(callback_dir)
        model_name = self.model_params["model_type"] + '_model_' + str(self.date.day) + '_' + str(self.date.month) + '_' + str(
            self.date.year) + '.{epoch:03d}.h5'

        if not os.path.isdir(callback_dir):
            os.makedirs(callback_dir)
        filepath = os.path.join(callback_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                                       patience=self.model_params["early_stopping_patience"],
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
                                                        batch_size=self.model_params["batch_size"], subset="training",
                                                        shuffle=self.model_params["shuffle_bool"])
            val_it = self.datagen.flow_from_directory(os.getcwd()+'/tiny-imagenet-200/train',
                                                      batch_size=self.model_params["batch_size"],
                                                      subset="validation", shuffle=self.model_params["shuffle_bool"])
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            # print(number_of_train_samples)
            # print(number_of_val_samples)

        else:

            # self.model_params["shuffle_training_data_bool"] = input("Do you want to shuffle the training data? (yes/no):  ")

            if self.model_params["shuffle_training_data_bool"]:
                self.model_params["shuffle_training_data_seed"] = input("Enter random seed (int):  ")
            else:
                self.model_params["shuffle_training_data_seed"] = float('inf')

            if self.model_params["dataset"] == "cifar10":
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                num_classes = len(classes)
                (training_images, training_labels), (
                validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if self.model_params["shuffle_training_data_seed"]!=float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.model_params["dataset"] == "mnist":
                num_classes=10
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if self.model_params["shuffle_training_data_seed"] != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.model_params["dataset"] == "cifar100":
                num_classes=100
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.cifar100.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if self.model_params["shuffle_training_data_seed"] != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            training_labels = to_categorical(training_labels, num_classes)
            validation_labels = to_categorical(validation_labels, num_classes)
            self.datagen.fit(train_X)

            train_it=self.datagen.flow(train_X, training_labels, batch_size=self.model_params["batch_size"])
            val_it = (valid_X, validation_labels)

            self.x_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it


    def get_iterations_random_sampling(self):

        e=[0,0]

        used_up_data=0

        for i in range(0,self.model_params["first_data_samples"]):
          self.y=self.y_train[:self.model_params["first_data_samples"]]
          self.x=self.x_train[:self.model_params["first_data_samples"]]

        while (e[1] < self.goal / 100)&(used_up_data + self.jump < self.x_train.shape[0]):

            for i in range(used_up_data,self.jump+used_up_data):
                self.x.append(self.x_train[self.random_num[i]])
                self.y.append(self.y_train[self.random_num[i]])

            used_up_data=used_up_data+self.jump

            h=self.model.fit(self.datagen.flow(np.array(self.x), np.array(self.y), batch_size=self.batch_size), validation_data=self.val_it,
                                     epochs=self.model_params["epochs"], verbose=1, workers=4,
                                     callbacks=self.callbacks)

            self.model_sampling_data["history"].append(h)

            self.model_sampling_data["accuracy"].append(self.model.evaluate(self.X_test, self.y_test))
            print(self.model_sampling_data["accuracy"][-1])


    def get_entropy(self,y_predicted):
        entropy_array = []

        for i in range(0,y_predicted.shape[0]):
            entropy_sum = 0
            for j in range(0,y_predicted.shape[1]):
              if y_predicted[i][j] == 0:
                continue
              single_entropy = y_predicted[i][j]*math.log(y_predicted[i][j])
              entropy_sum-=single_entropy
            entropy_array.append([int(i), entropy_sum])

        entropy_array.sort(key=lambda x:x[1],reverse=True)

        return entropy_array

    def get_least_confidence(self,y_predicted):
        probability_array = []

        for i in range(0,y_predicted.shape[0]):
            probability_array.append([int(i),np.max(y_predicted[i])])

        probability_array.sort(key=lambda x:x[1])

        return probability_array

    def get_ratio(self,y_predicted):
        ratio_array = []

        for i in range(0,y_predicted.shape[0]):
            y=y_predicted[i]
            sorted(y,reverse=True)
            first_max=y[0]
            second_max=y[1]
            if second_max==0:
              continue
            ratio_array.append([int(i),first_max/second_max])

        ratio_array.sort(key=lambda x:x[1])

        return ratio_array


    def get_margin(self,y_predicted):
        b=[]
        for i in range(0,y_predicted.shape[0]):
            a=y_predicted[i]
            c=sorted(a,reverse=True)
            difference=c[0]-c[1]

            b.append([int(i),difference])

        b.sort(key=lambda x:x[1])

        return b

    def get_high_confidence(self,y_predicted):
        probability_array = []

        for i in range(0,y_predicted.shape[0]):
            probability_array.append([int(i),np.max(y_predicted[i])])

        probability_array.sort(key=lambda x:x[1],reverse=True)

        return probability_array

    # def load_trained_model(self):

    def initial_training(self):

        print("Training...")

        if self.model_params["dataset"] == "tinyimagenet":
            self.model = self.define_compile_model()
            self.model_params["model_summary"] = self.model.summary()

            self.model.summary()

            history = self.model.fit(self.train_it, validation_data=self.val_it,
                                     epochs=self.model_params["epochs"], verbose=1, workers=4,
                                     callbacks=self.callbacks)

        else:

            self.model = self.define_compile_model()
            self.model_params["model_summary"] = self.model.summary()

            self.model.summary()

            history = self.model.fit(self.datagen.flow(self.x_train[:self.model_params["first_data_samples"]], self.y_train[:self.model_params["first_data_samples"]], batch_size=self.model_params["batch_size"]), validation_data=(self.X_test, self.y_test),
                                     epochs=self.model_params["epochs"], verbose=1, workers=4,
                                     callbacks=self.callbacks)

        self.model.save(self.training_dir)

        with open(self.training_dir + f'/initial_training_history', 'wb') as handle:
            pickle.dump(history.history, handle)
        print("Saved.")

    def bayesian(self, least_confidence_coeff):

        x_temp = copy.deepcopy(self.x)
        y_temp = copy.deepcopy(self.y)

        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=self.model_params["lr_reducer_patience"],
                                       min_lr=0.5e-6)

        params = {'least_confidence_coeff': least_confidence_coeff}
        high_confidence_coeff = 1 - least_confidence_coeff

        temp_model = load_model(os.getcwd() + "/model_for_bayes_optimization.h5")

        y_predicted = temp_model.predict(self.x_train_remaining)

        values_lc = self.get_least_confidence(y_predicted)
        values_hc = self.get_high_confidence(y_predicted)

        print(f"least_confidence_coeff: {least_confidence_coeff}, high_confidence_coeff: {high_confidence_coeff}")

        least_confidence_jump = int(self.jump * least_confidence_coeff)
        high_confidence_jump = self.jump - least_confidence_jump

        for i in range(0, least_confidence_jump):
            x_temp.append(self.x_train_remaining[values_lc[i][0]])
            y_temp.append(self.y_train_remaining[values_lc[i][0]])

        for i in range(0, high_confidence_jump):
            x_temp.append(self.x_train_remaining[values_hc[i][0]])
            y_temp.append(self.y_train_remaining[values_hc[i][0]])

        h = temp_model.fit(datagen.flow(np.array(x_temp), np.array(y_temp), batch_size=self.batch_size),
                           validation_data=(self.X_test, self.y_test), epochs=50, callbacks=[lr_scheduler,lr_reducer], verbose=1,
                           workers=4)

        score = temp_model.evaluate(self.X_test, self.y_test)

        return 100 * float(score[1])

    def get_bayes_coeff(self):

        params = {'least_confidence_coeff': (0, 1)}
        bo = BayesianOptimization(self.bayesian, params, random_state=22)
        bo.maximize(init_points=20, n_iter=10)

        params = bo.max['params']
        least_confidence_coeff = params["least_confidence_coeff"]

        return least_confidence_coeff

    def get_iterations(self):

        if self.model_params["sampling_type"] == "random":
          return self.get_iterations_random_sampling()

        if self.model_params["load_saved_model_bool"]:
            self.x_train_remaining=self.model_sampling_data["x_train"]
            self.y_train_remaining = self.model_sampling_data["y_train"]
            self.x=self.model_sampling_data["x"]
            self.y = self.model_sampling_data["y"]
        else:
            self.x_train_remaining=copy.deepcopy(self.x_train[self.model_params["first_data_samples"]:self.x_train.shape[0]])
            self.y_train_remaining=copy.deepcopy(self.y_train[self.model_params["first_data_samples"]:self.x_train.shape[0]])

            self.y=self.y_train[:self.model_params["first_data_samples"]]
            self.x=self.x_train[:self.model_params["first_data_samples"]]

        while (self.model_sampling_data["accuracy"][-1][1] < self.model_params["goal"] / 100)&(self.model_params["jump"] <= self.x_train_remaining.shape[0]):

            total_index=[*range(0, self.x_train_remaining.shape[0], 1)]
            y_predicted = self.model.predict(self.x_train_remaining)
            index = []

            if self.model_sampling_data["timestamps"]:
                print(f"Avg expected time for the iteration: {sum(self.timestamps)//len(self.timestamps)} seconds...")
            else:
                print(f"Avg expected time for the iteration: INFINITE seconds...")

            start = time.time()

            if self.model_params["sampling_type"] == "margin":
              values = self.get_margin(y_predicted)
            elif self.model_params["sampling_type"] == "leastconfidence":
              values = self.get_least_confidence(y_predicted)
            elif self.model_params["sampling_type"] == "highestconfidence":
              values = self.get_high_confidence(y_predicted)
            elif self.model_params["sampling_type"] == "entropy":
              values = self.get_entropy(y_predicted)
            elif self.model_params["sampling_type"] == "ratio":
              values = self.get_ratio(y_predicted)
            elif self.model_params["sampling_type"] == "mixed":
                values_lc = self.get_least_confidence(y_predicted)
                values_hc = self.get_high_confidence(y_predicted)
                for i in range(0, self.jump // 2):
                    index.append(values_lc[i][0])
                    self.x.append(self.x_train_remaining[values_lc[i][0]])
                    self.y.append(self.y_train_remaining[values_lc[i][0]])

                    index.append(values_hc[i][0])
                    self.x.append(self.x_train_remaining[values_hc[i][0]])
                    self.y.append(self.y_train_remaining[values_hc[i][0]])

            elif self.model_params["sampling_type"]== "mixedbayes":
                self.model.save(os.getcwd() + "/model_for_bayes_optimization.h5")

                least_confidence_coeff = self.get_bayes_coeff()

                least_confidence_jump = int(self.jump * least_confidence_coeff)
                high_confidence_jump = self.jump - least_confidence_jump

                self.model_sampling_data["least_confidence_jump"].append(least_confidence_jump)
                self.model_sampling_data["high_confidence_jump"].append(high_confidence_jump)

                values_lc = self.get_least_confidence(y_predicted)
                values_hc = self.get_high_confidence(y_predicted)

                for i in range(0, least_confidence_jump):
                    index.append(values_lc[i][0])
                    self.x.append(self.x_train_remaining[values_lc[i][0]])
                    self.y.append(self.y_train_remaining[values_lc[i][0]])

                for i in range(0, high_confidence_jump):
                    index.append(values_hc[i][0])
                    self.x.append(self.x_train_remaining[values_hc[i][0]])
                    self.y.append(self.y_train_remaining[values_hc[i][0]])
            else:
              print("Invalid Input!")

            if self.model_params["sampling_type"] !="mixed" and self.model_params["sampling_type"] !="mixedbayes":
              for i in range(0,self.jump):
                index.append(values[i][0])
                self.x.append(self.x_train_remaining[values[i][0]])
                self.y.append(self.y_train_remaining[values[i][0]])

            remaining_indices = [element for element in total_index if element not in index]

            self.x_train_remaining = self.x_train_remaining[remaining_indices]
            self.y_train_remaining = self.y_train_remaining[remaining_indices]

            h=self.model.fit(self.datagen.flow(np.array(self.x), np.array(self.y), batch_size=self.model_params["batch_size"] ), validation_data=self.val_it,
                                     epochs=self.model_params["epoch"], verbose=1, workers=4,
                                     callbacks=self.callbacks)

            end = time.time()

            self.model_sampling_data["history"].append(h)

            self.model_sampling_data["accuracy"].append(self.model.evaluate(self.X_test, self.y_test))
            print(self.model_sampling_data["accuracy"][-1])

            # a=[]
            # for element in total_index:
            #   if element not in index:
            #     a.append(element)
            #
            # self.x_train_remaining=self.x_train_remaining[a]
            # self.y_train_remaining=self.y_train_remaining[a]

            self.model_sampling_data["timestamps"].append(end-start)
            self.model_sampling_data["x_train"] = self.x_train_remaining
            self.model_sampling_data["y_train"]=self.y_train_remaining
            self.model_sampling_data["x"]=self.x
            self.model_sampling_data["y"] = self.y

            if self.model_params["save_and_reload"]:
                with open(self.save_and_reload_dir+'/model_sampling_data.pickle', 'wb') as handle:
                    pickle.dump(self.model_sampling_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                self.model.save(self.save_and_reload_dir+'/latest_model')

        self.model.save(self.training_dir+f"/{self.date}")

        with open(self.training_dir + '/model_params.pickle', 'wb') as handle:
            pickle.dump(self.model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

class HTMLparser:
    def __init__(self, words):
        warnings.filterwarnings("ignore")

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
    def __init__(self, load=False, user_input=False, dataset="tinyimagenet", model_type="efficientnetb6", sampling_type="none", threshold=0.2, validation_split=0.3, batch_size=16, shuffle_bool=True):
        warnings.filterwarnings("ignore")

        self.load=load
        self.user_input=user_input
        self.dataset = dataset.lower()
        self.model_type = model_type.lower()
        self.sampling_type = sampling_type
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
            self.contextmodel=load_model(os.getcwd()+"/"+path)
        else:
            # check=False
            # for file in os.listdir(os.getcwd()):
            #     if file.endswith(f'{self.model_type}_{self.dataset}_{self.sampling_type}.zip'):
            #         zip_file = ZipFile(file)
            #         zip_file.extractall(os.getcwd() + f'/models_to_load/')
            #         self.contextmodel = load_model(os.getcwd() + f'/models_to_load/'+f'{self.model_type}_{self.dataset}_{self.sampling_type}')
            #         check=True
            # if not check:
            pretrained = PretrainedModel(model_type=self.model_type, dataset=self.dataset,
                                          sampling_type=self.sampling_type)
            self.contextmodel = pretrained.get_model()

        if self.user_input:
            path=input("Please enter image path in the current directory (str):    ")
            img = Image.open(os.getcwd() + f"/{path}")
            img.show()

            img = load_img(os.getcwd() + f"/{path}")
            img = img.resize((32, 32))
            img = img_to_array(img)

            img = img.reshape(1, 32, 32, 3)
            # img = iio.imread(os.getcwd()+f"{path}")
            self.pred = self.contextmodel.predict_generator(img)
        else:
            # random.randint(1, 3000)
            # np.ceil(self.number_of_val_samples / self.batch_size)
            # , steps = self.batch_size
            self.pred = self.contextmodel.predict_generator(self.val_it,1)

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


    def decide_context(self):

        from classes import i2d

        predicted_list = []
        prediction_index = []
        final_classes = []

        output = []

        labels = self.combined_it.class_indices
        labels2 = dict((v, k) for k, v in labels.items())

        # for i in range(0, self.pred.shape[0]):
        #     classes_prob_list = []
        #     for j in range(0, self.output_layer_classes):
        #         classes_prob_list.append([int(j), self.pred[i][j]])
        #
        #     classes_prob_list.sort(key=lambda x: x[1], reverse=True)
        #
        #     first_highest_predicted_classes = classes_prob_list[0][0]
        #     first_highest_predicted_class_confidence = classes_prob_list[0][1]
        #
        #     second_highest_predicted_classes = classes_prob_list[1][0]
        #     second_highest_predicted_class_confidence = classes_prob_list[1][1]
        #
        #     predicted_list.append([first_highest_predicted_classes, first_highest_predicted_class_confidence,
        #                            second_highest_predicted_classes, second_highest_predicted_class_confidence])
        #
        # if (predicted_list[0][1] - predicted_list[0][3]) >= self.threshold:
        #     prediction_index.append(predicted_list[0][0])
        # elif (predicted_list[0][1] - predicted_list[0][3]) < self.threshold:
        #     prediction_index.append(predicted_list[0][0])
        #     prediction_index.append(predicted_list[0][2])

        for i in range(0, len(prediction_index)):
            final_classes.append(i2d[labels2[prediction_index[i]]])

        for ele in final_classes:
            b = ele.split(',')
            output = output + b

        return output

    def download_dataset(self):

        print("Extracting...")
        if not 'tiny-imagenet-200' in os.listdir(os.getcwd()):
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
                                                        shuffle=self.shuffle_bool,seed=random.randint(1,100))
            val_it = self.datagen.flow_from_directory(os.getcwd() + '/tiny-imagenet-200/train',
                                                      batch_size=self.batch_size,
                                                      subset="validation", shuffle=self.shuffle_bool,seed=random.randint(1,100))

            self.combined_it = self.datagen.flow_from_directory(os.getcwd() + '/tiny-imagenet-200/train', batch_size=self.batch_size,)

            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            self.number_of_val_samples = len(val_filenames)
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

            self.x_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it


class QuestionAnswer:
    def __init__(self, context, chatbot="bert"):
        warnings.filterwarnings("ignore")

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


# https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
class TextTranslator:
    def __init__(self, file_bool=False,filepath="", translate_from_language="ben", translate_to_language="en", speak_bool=False):
        self.file_bool=file_bool
        self.filepath=filepath
        self.translate_from_language = translate_from_language
        self.translate_to_language = translate_to_language
        self.speak_bool = speak_bool

        if self.speak_bool:
            try:
                os.system("sudo apt install espeak")
                os.system("sudo apt install libespeak-dev")
                os.system("pip install pyaudio")
                os.system("sudo apt install python3-pyaudio")
                os.system("sudo apt install portaudio19 - dev")

            except:
                os.system("!sudo apt install espeak")
                os.system("!sudo apt install libespeak-dev")
                os.system("!pip install pyaudio")
                os.system("!sudo apt install python3-pyaudio")
                os.system("!sudo apt install portaudio19 - dev")

    def speak(self, command):
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

    def translate(self):

        if self.file_bool:
            file = open(os.getcwd()+f"/{self.filepath}","r")
            text=file.readlines()[0]
            print(file.readlines())
        else:
            text=input("What do you want to translate?:     ")

        translator = Translator()

        k = translator.translate(text, dest=self.translate_to_language)
        with open(f'{self.imgpath}_text_{self.translate_to_language}.txt', mode='w') as file:
            file.write(k.text)
        print(k.text)

        if self.speak_bool:
            self.speak(k.text)


class ImageTranslator:
    def __init__(self, imgpath, translate_from_language="ben", translate_to_language="en", speak_bool=False):
        self.translate_from_language=translate_from_language
        self.translate_to_language=translate_to_language
        self.imgpath=imgpath
        self.speak_bool=speak_bool



        if self.speak_bool:
            try:
                os.system("sudo apt install espeak")
                os.system("sudo apt install libespeak-dev")
                os.system("pip install pyaudio")
                os.system("sudo apt install python3-pyaudio")
                os.system("sudo apt install portaudio19 - dev")

            except:
                os.system("!sudo apt install espeak")
                os.system("!sudo apt install libespeak-dev")
                os.system("!pip install pyaudio")
                os.system("!sudo apt install python3-pyaudio")
                os.system("!sudo apt install portaudio19 - dev")

    def speak(self, command):

        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

    def translate(self):

        try:
            os.system("sudo apt install tesseract-ocr")
            os.system("apt install libtesseract-dev")
            os.system(f"apt install tesseract-ocr-{self.translate_from_language}")
            os.system(f"apt install tesseract-ocr-{self.translate_to_language}")
        except:
            os.system("!sudo apt install tesseract-ocr")
            os.system("!apt install libtesseract-dev")
            os.system(f"!apt install tesseract-ocr-{self.translate_from_language}")
            os.system(f"!apt install tesseract-ocr-{self.translate_to_language}")

        img = Image.open(os.getcwd()+"/"+self.imgpath)

        result = pytesseract.image_to_string(img,lang=self.translate_from_language)
        with open(f'{self.imgpath}_text_{self.translate_from_language}.txt', mode='w') as file:
            file.write(result)
            print(result)

        translator = Translator()

        k = translator.translate(result.replace("\n"," ")[:-5], dest=self.translate_to_language)
        with open(f'{self.imgpath}_text_{self.translate_to_language}.txt', mode='w') as file:
            file.write(k.text)
        print(k.text)

        if self.speak_bool:
            self.speak(k.text)


class Conversation:
    def __init__(self):
        warnings.filterwarnings("ignore")

        self.conversational_pipeline = pipeline("conversational")

        try:
            os.system("apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg")
        except:
            os.system("!apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg")

        self.recognizer = sr.Recognizer()

        self.RECORD = """
                const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
                const b2text = blob => new Promise(resolve => {
                  const reader = new FileReader()
                  reader.onloadend = e => resolve(e.srcElement.result)
                  reader.readAsDataURL(blob)
                })
                var record = time => new Promise(async resolve => {
                  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
                  recorder = new MediaRecorder(stream)
                  chunks = []
                  recorder.ondataavailable = e => chunks.push(e.data)
                  recorder.start()
                  await sleep(time)
                  recorder.onstop = async ()=>{
                    blob = new Blob(chunks)
                    text = await b2text(blob)
                    resolve(text)
                  }
                  recorder.stop()
                })
                """
    def record(self,sec=3):
        display(Javascript(RECORD))
        sec += 1
        s = output.eval_js('record(%d)' % (sec*1000))
        b = b64decode(s.split(',')[1])
        return b

    def speak(self,command):

        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

    def answer(self,question):
        output = self.conversational_pipeline([question])
        out_list = str(output).split("\n")
        temp = out_list[2].split(">>")
        output = temp[1].replace("\n", "")
        return output

    def converse(self):

        while True:

            try:

                audio_source = sr.AudioData(self.record(), 16000, 2)

                question = self.recognize_google(audio_data=audio_source,language = 'en-IN')
                question = question.lower()

                print(f"{question}?")
                # self.speak(question)

                answer_output=self.answer(question)
                self.speak(answer_output)
                #
                # with sr.Microphone() as source2:
                #
                #     r.adjust_for_ambient_noise(source2,duration=5)
                #     print("Listening...")
                #
                #     audio2 = r.listen(source2,timeout=5,phrase_time_limit=5)
                #
                #     print("Recognizing...")
                #
                #     question = self.recognizer.recognize_google(audio2,language = 'en-IN')
                #     question = question.lower()
                #
                #     print("Did you say " + question)
                #     self.speak(question)
                #
                #     answer_output=self.answer(question)
                #     self.speak(answer_output)

            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

            except sr.UnknownValueError:
                print("Unknown Error Occured")


class PDFtoText:
    def __init__(self, pdfpath, speak_bool=False):
        self.pdfpath = imgpath
        self.speak_bool = speak_bool

    def convert(self, command):

        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

        if speak:
            try:
                os.system("sudo apt install espeak")
                os.system("sudo apt install libespeak-dev")
            except:
                os.system("!sudo apt install espeak")
                os.system("!sudo apt install libespeak-dev")

        path = open(os.getcwd() + f'/{self.pdfpath}', 'rb')

        pdfReader = PyPDF2.PdfFileReader(path)

        output = ""
        for i in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(i)
            output += pageObj.extractText()

        with open(f'{self.pdfpath}_text.txt', mode='w') as file:
            file.write(output)

        print(output)

        if self.speak_bool:
            self.speak(k.text)

        return output


class Clustering:
    def __init__(self, data, score_type="silhouette", pca_plot=False, type="kmeans"):

        self.data = data.dropna(axis=1)

        self.pca_plot = pca_plot
        self.type = type
        self.score_type = score_type
        self.n_components = self.get_n_components()

        print(f"no. of components: {self.n_components}")
        self.preprocessor = Pipeline([("scaler", MinMaxScaler()), ("pca", PCA())])
        self.preprocessor.fit(self.data)
        self.preprocessed_data = self.preprocessor.transform(self.data)

        if self.type == "kmeans":
            self.kmeans_kwargs = {"init": "random", "n_init": 50, "max_iter": 500, "random_state": 22, }
            self.n_clusters = self.get_n_clusters()
            self.kmeans = KMeans(n_clusters=self.n_clusters, **self.kmeans_kwargs)
            self.kmeans.fit(self.preprocessed_data)
            self.labels = self.kmeans.labels_

        elif self.type == "spectral":
            self.spectral_kwargs = {"n_init": 50, "random_state": 22, "affinity": 'nearest_neighbors', }
            # 'eigen_solver':"arpack",
            self.n_clusters = self.get_n_clusters()
            self.spectral = SpectralClustering(n_clusters=self.n_clusters, **self.spectral_kwargs)
            self.spectral.fit(self.preprocessed_data)
            self.labels = self.spectral.labels_

        elif self.type == "heirarchical":
            self.heirarchical_kwargs = {"metric": 'euclidean', "linkage": 'ward'}
            self.n_clusters = self.get_n_clusters()
            self.heirarchical = AgglomerativeClustering(n_clusters=self.n_clusters, **self.heirarchical_kwargs)
            self.heirarchical.fit(self.preprocessed_data)
            self.labels = self.heirarchical.labels_

        print(f"no. of clusters: {self.n_clusters}")

    def plot_groups(self):
        fte_colors = {
            -1: "#003428",
            0: "#008fd5",
            1: "#fc4f30",
            2: "#000000",
            3: "#ffffff",
            4: "#389241",
            5: "#434822"}

        if self.type == "kmeans":
            a = self.kmeans.fit_predict(self.preprocessed_data)
            fig, ax = plt.subplots()
            sns.scatterplot(x=self.preprocessed_data[:, 0], y=self.preprocessed_data[:, 1], hue=a, ax=ax)
            kmeans_silhouette = silhouette_score(self.preprocessed_data, self.kmeans.labels_).round(2)
            ax.set(title=f"{self.type} Clustering:    Silhouette: {kmeans_silhouette}")

        elif self.type == "spectral":
            a = self.spectral.fit_predict(self.preprocessed_data)
            fig, ax = plt.subplots()
            sns.scatterplot(x=self.preprocessed_data[:, 0], y=self.preprocessed_data[:, 1], hue=a, ax=ax)
            spectral_silhouette = silhouette_score(self.preprocessed_data, self.spectral.labels_).round(2)
            ax.set(title=f"{self.type} Clustering:    Silhouette: {spectral_silhouette}")

        elif self.type == "heirarchical":
            a = self.heirarchical.fit_predict(self.preprocessed_data)
            fig, ax = plt.subplots()
            sns.scatterplot(x=self.preprocessed_data[:, 0], y=self.preprocessed_data[:, 1], hue=a, ax=ax)
            heirarchical_silhouette = silhouette_score(self.preprocessed_data, self.heirarchical.labels_).round(2)
            ax.set(title=f"{self.type} Clustering:    Silhouette: {heirarchical_silhouette}")

        else:
            print("Invalid Input!")

    def get_n_components(self):
        pca = PCA(random_state=22)

        x_pca = pca.fit_transform(self.data)

        exp_var_pca = pca.explained_variance_ratio_

        cum_sum_eigenvalues = np.cumsum(exp_var_pca)

        n = -1

        for i in range(len(cum_sum_eigenvalues)):
            if cum_sum_eigenvalues[i] > 0.90:
                n = i
                break

        if n == -1:
            for i in range(len(cum_sum_eigenvalues)):
                if cum_sum_eigenvalues[i] > 0.85:
                    n = i
                    break

        if self.pca_plot:
            plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
                    label='Individual explained variance')
            plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
                     label='Cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal component index')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        return n

    def get_n_clusters(self):

        coeff = []

        if self.score_type == "silhouette":
            if self.type == "kmeans":
                for k in range(2, 11):
                    kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
                    kmeans.fit(self.preprocessed_data)
                    score = silhouette_score(self.preprocessed_data, kmeans.labels_)
                    coeff.append(score)
            elif self.type == "spectral":
                for k in range(2, 11):
                    spectral = SpectralClustering(n_clusters=k, **self.spectral_kwargs)
                    spectral.fit(self.preprocessed_data)
                    score = silhouette_score(self.preprocessed_data, spectral.labels_)
                    coeff.append(score)
            elif self.type == "heirarchical":
                for k in range(2, 11):
                    heirarchical = AgglomerativeClustering(n_clusters=k, **self.heirarchical_kwargs)
                    heirarchical.fit(self.preprocessed_data)
                    score = silhouette_score(self.preprocessed_data, heirarchical.labels_)
                    coeff.append(score)

            plt.style.use("fivethirtyeight")
            plt.plot(range(2, 11), coeff)
            plt.xticks(range(2, 11))
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            plt.show()

        elif self.score_type == "sse":
            if self.type == "kmeans":
                for k in range(2, 11):
                    kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
                    kmeans.fit(self.preprocessed_data)
                    coeff.append(kmeans.inertia_)
            elif self.type == "spectral":
                for k in range(2, 11):
                    spectral = SpectralClustering(n_clusters=k, **self.spectral_kwargs)
                    spectral.fit(self.preprocessed_data)
                    coeff.append(spectral.inertia_)
            elif self.type == "heirarchical":
                for k in range(2, 11):
                    heirarchical = AgglomerativeClustering(n_clusters=k, **self.heirarchical_kwargs)
                    heirarchical.fit(self.preprocessed_data)
                    coeff.append(heirarchical.inertia_)

            plt.style.use("fivethirtyeight")
            plt.plot(range(2, 11), coeff)
            plt.xticks(range(2, 11))
            plt.xlabel("Number of Clusters")
            plt.ylabel("SSE")
            plt.show()

        kl = KneeLocator(range(2, 11), coeff, curve="convex", direction="decreasing")
        print(kl.elbow)

        return kl.elbow






