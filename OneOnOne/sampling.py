import warnings
import random
import pickle
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



class Sampling:
    def __init__(self, file_input_params=False, filepath="", sampling_type="mixedbayes", dataset="cifar10",
                 model_type="resnet50", goal=99, jump=5000, validation_split=0.3, first_data_samples=10000,
                 batch_size=16, epochs=250, shuffle_bool=True, early_stopping_patience=10, lr_reducer_patience=10,
                 save_and_reload_bool=True, load_saved_model_bool=False, shuffle_training_data_bool=False,
                 shuffle_training_data_seed=float('inf')):
        warnings.filterwarnings("ignore")

        self.model_params = {}

        numeric_args = ["goal", "jump", "first_data_samples", "batch_size", "epochs", "early_stopping_patience",
                        "lr_reducer_patience", "shuffle_training_data_seed"]
        boolean_args = ["shuffle_bool", "save_and_reload_bool", "load_saved_model_bool", "shuffle_training_data_bool"]
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
                    self.model_params[var] = int(val)
                elif var in float_args:
                    self.model_params[var] = float(val)
                elif var in boolean_args:
                    if val == "True":
                        self.model_params[var] = True
                    else:
                        self.model_params[var] = False
                else:
                    self.model_params[var] = val

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

        self.training_dir = os.getcwd() + f"/trained_models/" + str(self.model_params["model_type"]) + "/" + str(
            self.model_params["dataset"]) + "/" + str(self.model_params["sampling_type"]) + "_" + str(
            self.model_params["first_data_samples"]) + "k"

        if not os.path.isdir(self.training_dir):
            os.makedirs(self.training_dir)

        with open(self.training_dir + '/model_params.pickle', 'wb') as handle:
            pickle.dump(self.model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.callbacks = self.get_callbacks()
        self.datagen = self.get_datagen()

        self.train_it, self.val_it = self.get_dataset()

        if self.model_params["save_and_reload_bool"]:
            self.save_and_reload_dir = os.getcwd() + f'/reload_params/{self.model_params["model_type"]}/{self.model_params["dataset"]}/{self.model_params["sampling_type"]}_{self.model_params["first_data_samples"]}k'

            if not os.path.isdir(self.save_and_reload_dir):
                os.makedirs(self.save_and_reload_dir)

        if not self.model_params["load_saved_model_bool"]:
            self.model_sampling_data = {"accuracy": [[0, 0]], "timestamps": [], "history": [], "x_train": [],
                                        "y_train": [], "x": [], "y": []}
            if self.model_params["sampling_type"] == "mixedbayes":
                self.model_sampling_data["least_confidence_jump"] = []
                self.model_sampling_data["high_confidence_jump"] = []

            # with open(self.save_and_reload_dir + '/model_params.pickle', 'wb') as handle:
            #     pickle.dump(self.model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #
            # with open(self.save_and_reload_dir + '/model_sampling_data.pickle', 'wb') as handle:
            #     pickle.dump(self.model_sampling_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(self.save_and_reload_dir + '/model_sampling_data.pickle', 'rb') as handle:
                self.model_sampling_data = pickle.load(handle)

            with open(self.save_and_reload_dir + '/model_params.pickle', 'rb') as handle:
                self.model_params = pickle.load(handle)

            self.model = load_model(self.save_and_reload_dir + '/latest_model')

        if self.model_params["sampling_type"] == "random":
            random_num = np.random.randint(self.model_params["first_data_samples"], self.x_train.shape[0],
                                           size=self.x_train.shape[0] - self.model_params["first_data_samples"])
            self.random_num = random_num

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
        x = tf.keras.layers.Dense(self.model_params["output_layer_classes"], activation="softmax",
                                  name="classification")(x)
        return x

    def feature_extractor(self, inputs):

        if self.model_params["model_type"] == "resnet50":
            feature_extractor = tf.keras.applications.resnet50.ResNet50(input_shape=self.model_params["input_shape"],
                                                                        include_top=False,
                                                                        weights='imagenet',
                                                                        classes=self.model_params[
                                                                            "output_layer_classes"],
                                                                        classifier_activation="softmax")(inputs)
        elif self.model_params["model_type"] == "efficientnetb6":
            feature_extractor = tensorflow.keras.applications.EfficientNetB6(
                input_shape=self.model_params["input_shape"],
                include_top=False,
                weights='imagenet',
                classes=self.model_params["output_layer_classes"],
                classifier_activation="softmax")(inputs)

        elif self.model_params["model_type"] == "densenet":
            feature_extractor = tf.keras.applications.DenseNet201(input_shape=self.model_params["input_shape"],
                                                                  include_top=False,
                                                                  weights='imagenet',
                                                                  classes=self.model_params["output_layer_classes"],
                                                                  classifier_activation="softmax")(inputs)
        elif self.model_params["model_type"] == "vgg19":
            feature_extractor = tf.keras.applications.VGG19(input_shape=self.model_params["input_shape"],
                                                            include_top=False,
                                                            weights='imagenet',
                                                            classes=self.model_params["output_layer_classes"],
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
        inputs = tf.keras.layers.Input(shape=(32, 32, 3))

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

        callback_dir = os.getcwd() + f'/saved_callbacks/{self.model_params["model_type"]}/{self.model_params["dataset"]}/{self.model_params["sampling_type"]}_{self.model_params["first_data_samples"]}k'

        if not os.path.isdir(callback_dir):
            os.makedirs(callback_dir)
        model_name = self.model_params["model_type"] + '_model_' + str(self.date.day) + '_' + str(
            self.date.month) + '_' + str(
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
            train_it = self.datagen.flow_from_directory(os.getcwd() + '/tiny-imagenet-200/train',
                                                        batch_size=self.model_params["batch_size"], subset="training",
                                                        shuffle=self.model_params["shuffle_bool"])
            val_it = self.datagen.flow_from_directory(os.getcwd() + '/tiny-imagenet-200/train',
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

                if self.model_params["shuffle_training_data_seed"] != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.model_params["dataset"] == "mnist":
                num_classes = 10
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if self.model_params["shuffle_training_data_seed"] != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.model_params["dataset"] == "cifar100":
                num_classes = 100
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

            train_it = self.datagen.flow(train_X, training_labels, batch_size=self.model_params["batch_size"])
            val_it = (valid_X, validation_labels)

            self.x_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it

    def get_iterations_random_sampling(self):

        e = [0, 0]

        used_up_data = 0

        for i in range(0, self.model_params["first_data_samples"]):
            self.y = self.y_train[:self.model_params["first_data_samples"]]
            self.x = self.x_train[:self.model_params["first_data_samples"]]

        while (e[1] < self.goal / 100) & (used_up_data + self.jump < self.x_train.shape[0]):

            for i in range(used_up_data, self.jump + used_up_data):
                self.x.append(self.x_train[self.random_num[i]])
                self.y.append(self.y_train[self.random_num[i]])

            used_up_data = used_up_data + self.jump

            h = self.model.fit(self.datagen.flow(np.array(self.x), np.array(self.y), batch_size=self.batch_size),
                               validation_data=self.val_it,
                               epochs=self.model_params["epochs"], verbose=1, workers=4,
                               callbacks=self.callbacks)

            self.model_sampling_data["history"].append(h)

            self.model_sampling_data["accuracy"].append(self.model.evaluate(self.X_test, self.y_test))
            print(self.model_sampling_data["accuracy"][-1])

    def get_entropy(self, y_predicted):
        entropy_array = []

        for i in range(0, y_predicted.shape[0]):
            entropy_sum = 0
            for j in range(0, y_predicted.shape[1]):
                if y_predicted[i][j] == 0:
                    continue
                single_entropy = y_predicted[i][j] * math.log(y_predicted[i][j])
                entropy_sum -= single_entropy
            entropy_array.append([int(i), entropy_sum])

        entropy_array.sort(key=lambda x: x[1], reverse=True)

        return entropy_array

    def get_least_confidence(self, y_predicted):
        probability_array = []

        for i in range(0, y_predicted.shape[0]):
            probability_array.append([int(i), np.max(y_predicted[i])])

        probability_array.sort(key=lambda x: x[1])

        return probability_array

    def get_ratio(self, y_predicted):
        ratio_array = []

        for i in range(0, y_predicted.shape[0]):
            y = y_predicted[i]
            sorted(y, reverse=True)
            first_max = y[0]
            second_max = y[1]
            if second_max == 0:
                continue
            ratio_array.append([int(i), first_max / second_max])

        ratio_array.sort(key=lambda x: x[1])

        return ratio_array

    def get_margin(self, y_predicted):
        b = []
        for i in range(0, y_predicted.shape[0]):
            a = y_predicted[i]
            c = sorted(a, reverse=True)
            difference = c[0] - c[1]

            b.append([int(i), difference])

        b.sort(key=lambda x: x[1])

        return b

    def get_high_confidence(self, y_predicted):
        probability_array = []

        for i in range(0, y_predicted.shape[0]):
            probability_array.append([int(i), np.max(y_predicted[i])])

        probability_array.sort(key=lambda x: x[1], reverse=True)

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

            history = self.model.fit(self.datagen.flow(self.x_train[:self.model_params["first_data_samples"]],
                                                       self.y_train[:self.model_params["first_data_samples"]],
                                                       batch_size=self.model_params["batch_size"]),
                                     validation_data=(self.X_test, self.y_test),
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
                           validation_data=(self.X_test, self.y_test), epochs=50, callbacks=[lr_scheduler, lr_reducer],
                           verbose=1,
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
            self.x_train_remaining = self.model_sampling_data["x_train"]
            self.y_train_remaining = self.model_sampling_data["y_train"]
            self.x = self.model_sampling_data["x"]
            self.y = self.model_sampling_data["y"]
        else:
            self.x_train_remaining = copy.deepcopy(
                self.x_train[self.model_params["first_data_samples"]:self.x_train.shape[0]])
            self.y_train_remaining = copy.deepcopy(
                self.y_train[self.model_params["first_data_samples"]:self.x_train.shape[0]])

            self.y = self.y_train[:self.model_params["first_data_samples"]]
            self.x = self.x_train[:self.model_params["first_data_samples"]]

        while (self.model_sampling_data["accuracy"][-1][1] < self.model_params["goal"] / 100) & (
                self.model_params["jump"] <= self.x_train_remaining.shape[0]):

            total_index = [*range(0, self.x_train_remaining.shape[0], 1)]
            y_predicted = self.model.predict(self.x_train_remaining)
            index = []

            if self.model_sampling_data["timestamps"]:
                print(f"Avg expected time for the iteration: {sum(self.timestamps) // len(self.timestamps)} seconds...")
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

            elif self.model_params["sampling_type"] == "mixedbayes":
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

            if self.model_params["sampling_type"] != "mixed" and self.model_params["sampling_type"] != "mixedbayes":
                for i in range(0, self.jump):
                    index.append(values[i][0])
                    self.x.append(self.x_train_remaining[values[i][0]])
                    self.y.append(self.y_train_remaining[values[i][0]])

            remaining_indices = [element for element in total_index if element not in index]

            self.x_train_remaining = self.x_train_remaining[remaining_indices]
            self.y_train_remaining = self.y_train_remaining[remaining_indices]

            h = self.model.fit(
                self.datagen.flow(np.array(self.x), np.array(self.y), batch_size=self.model_params["batch_size"]),
                validation_data=self.val_it,
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

            self.model_sampling_data["timestamps"].append(end - start)
            self.model_sampling_data["x_train"] = self.x_train_remaining
            self.model_sampling_data["y_train"] = self.y_train_remaining
            self.model_sampling_data["x"] = self.x
            self.model_sampling_data["y"] = self.y

            if self.model_params["save_and_reload"]:
                with open(self.save_and_reload_dir + '/model_sampling_data.pickle', 'wb') as handle:
                    pickle.dump(self.model_sampling_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                self.model.save(self.save_and_reload_dir + '/latest_model')

        self.model.save(self.training_dir + f"/{self.date}")

        with open(self.training_dir + '/model_params.pickle', 'wb') as handle:
            pickle.dump(self.model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

