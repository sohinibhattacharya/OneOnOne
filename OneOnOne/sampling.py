class Sampling:
    def __init__(self, samplingtype, dataset="cifar10", model_type="resnet50", goal=99, jump=5000,
                 first_data_samples=10000, batch_size=16, epochs=250, shuffle_bool=True, early_stopping_patience=10,
                 lr_reducer_patience=10):

        self.validation_split = 0
        self.model_type = model_type.lower()
        self.epochs = epochs
        self.jump = jump
        self.samplingtype = samplingtype.lower()
        self.goal = goal
        self.shuffle_bool = shuffle
        self.batch_size = batch_size
        self.dataset = dataset.lower()
        self.output_layer_classes = 0
        self.input_shape = (32, 32, 3)
        self.date = datetime.datetime.now()
        self.first_data_samples = first_data_samples
        self.early_stopping_patience = early_stopping_patience
        self.lr_reducer_patience = lr_reducer_patience
        self.callbacks = self.get_callbacks()
        self.datagen = self.get_datagen()
        self.train_it, self.val_it = self.get_dataset()

        if self.dataset == "cifar10" or self.dataset == "mnist":
            self.output_layer_classes = 10
            self.input_shape = (224, 224, 3)


        elif self.dataset == "tinyimagenet":
            self.output_layer_classes = 200
            self.input_shape = (32, 32, 3)
            from classes import i2d
            self.download_dataset()

        else:
            print("Invalid Input!")

        if samplingtype == "random":
            random_num = np.random.randint(self.first_data_samples, self.X_train.shape[0],
                                           size=self.X_train.shape[0] - self.first_data_samples)
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

        save_dir = os.getcwd() + f'/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}'
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

        if self.dataset == "tinyimagenet":
            self.validation_split = input("Validation split (example - 0.3):   ")
        else:
            pass

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

        print("Done.")

    def get_dataset(self):

        if self.dataset == "tinyimagenet":
            train_it = self.datagen.flow_from_directory(os.getcwd() + 'tiny-imagenet-200/train',
                                                        batch_size=self.batch_size, subset="training",
                                                        shuffle=self.shuffle_bool)
            val_it = self.datagen.flow_from_directory(os.getcwd() + 'tiny-imagenet-200/train',
                                                      batch_size=self.batch_size,
                                                      subset="validation", shuffle=self.shuffle_bool)
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            print(number_of_train_samples)
            print(number_of_val_samples)

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

                if seed_token != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.dataset == "mnist":
                num_classes = 10
                (training_images, training_labels), (
                    validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

                train_X = self.preprocess_image_input(training_images)
                valid_X = self.preprocess_image_input(validation_images)

                if seed_token != float('inf'):
                    random.Random(seed_token).shuffle(train_X)
                    random.Random(seed_token).shuffle(training_labels)

            elif self.dataset == "cifar100":
                num_classes = 100
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

            train_it = self.datagen.flow(train_X, training_labels, batch_size=self.batch_size)
            val_it = (valid_X, validation_labels)

            self.X_train = train_X
            self.X_test = valid_X
            self.y_train = training_labels
            self.y_test = validation_labels

        return train_it, val_it

    def get_iterations_rs(self):

        e = [0, 0]

        k = 0

        num = 0

        y = []
        x = []
        for i in range(0, self.first_data_samples):
            y.append(self.y_train[i])
            x.append(self.X_train[i])

        while (e[1] < self.goal / 100) & (k + self.jump < self.X_train.shape[0]):

            for i in range(0 + k, self.jump + k):
                x.append(self.X_train[self.random_num[i]])
                y.append(self.y_train[self.random_num[i]])

            k = k + self.jump
            num = num + 1

            self.model.fit(np.array(x), np.array(y), epochs=50)

            e = self.model.evaluate(self.X_test, self.y_test)

        return e[1], num

    def get_entropy(self, y_predicted_en):
        sum_prob = 0
        entropy = []

        for i in range(0, y_predicted_en.shape[0]):
            for j in range(0, y_predicted_en.shape[1]):
                if (y_predicted_en[i][j] == 0):
                    continue
                k = y_predicted_en[i][j] * math.log(y_predicted_en[i][j])
                sum_prob = sum_prob + k
            entropy.append([int(i), -(sum_prob)])
            sum_prob = 0
            k = 0

        entropy.sort(key=lambda x: x[1], reverse=True)

        return entropy

    def get_lc(self, y_predicted_lc):
        probarray = []

        for i in range(0, y_predicted_lc.shape[0]):
            probarray.append([int(i), np.max(y_predicted_lc[i])])

        probarray.sort(key=lambda x: x[1])

        return probarray

    def get_ratio(self, y_predicted_r):
        ratio_array = []

        for i in range(0, y_predicted_r.shape[0]):
            y = y_predicted_r[i]
            sorted(y, reverse=True)
            first_max = y[0]
            second_max = y[1]
            if second_max == 0:
                continue
            ratio_array.append([int(i), first_max / second_max])

        ratio_array.sort(key=lambda x: x[1])

        return ratio_array

    def get_margin(self, y_predicted_margin):
        a = []
        b = []
        for i in range(0, y_predicted_margin.shape[0]):
            a = y_predicted_margin[i]
            c = sorted(a, reverse=True)
            difference = c[0] - c[1]

            b.append([int(i), difference])

        b.sort(key=lambda x: x[1])

        return b

    def get_hc(self, y_predicted_hc):
        probarray = []

        for i in range(0, y_predicted_hc.shape[0]):
            probarray.append([int(i), np.max(y_predicted_hc[i])])

        probarray.sort(key=lambda x: x[1], reverse=True)

        return probarray

    # def load_trained_model(self):

    def initial_training(self):

        print("Training...")

        if self.dataset == "tinyimagenet":
            self.model = self.define_compile_model()
            self.model.summary()

            history = self.model.fit(self.train_it, validation_data=self.val_it,
                                     epochs=self.epochs, verbose=1, workers=4,
                                     callbacks=self.callbacks)

            self.model.save(
                os.getcwd() + f"/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}_{self.first_data_samples}_{self.date}_initial_training")

            with open(f'{self.model_type}_{self.dataset}_{self.date}_history', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            print("Saved.")

        else:

            self.model = self.define_compile_model()
            self.model.summary()

            history = self.model.fit(self.datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size),
                                     validation_data=(self.X_test, self.y_test),
                                     epochs=self.epochs, verbose=1, workers=4,
                                     callbacks=self.callbacks)
            self.model.save(
                os.getcwd() + f"/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}_{self.first_data_samples}_{self.date}_initial_training")

            with open(f'{self.model_type}_{self.dataset}_{self.date}_history', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            print("saved.")

    def get_iterations(self):

        eval_metrics = [0, 0]

        num = 0
        batch_size_data = [0]
        acuracy_data = [0]
        epoch_data = []
        history_data = []

        y = []
        x = []

        if self.samplingtype == "random":
            eval_metrics[1], num = self.get_iterations_rs()

            return eval_metrics[1], num

        X_train_copy = self.X_train[self.first_data_samples:self.X_train.shape[0]]
        y_train_copy = self.y_train[self.first_data_samples:self.X_train.shape[0]]

        for i in range(0, self.first_data_samples):
            y.append(self.y_train[i])
            x.append(self.X_train[i])

        while (eval_metrics[1] < self.goal / 100) & (self.jump <= X_train_copy.shape[0]):

            total_index = [*range(0, X_train_copy.shape[0], 1)]
            y_predicted = self.model.predict(X_train_copy)

            if self.samplingtype == "margin":
                values = self.get_margin(y_predicted)
            elif self.samplingtype == "leastconfidence":
                values = self.get_lc(y_predicted)
            elif self.samplingtype == "highestconfidence":
                values = self.get_hc(y_predicted)
            elif self.samplingtype == "entropy":
                values = self.get_entropy(y_predicted)
            elif self.samplingtype == "ratio":
                values = self.get_ratio(y_predicted)
            elif self.samplingtype == "mixed":
                values_lc = self.get_lc(y_predicted)
                values_hc = self.get_hc(y_predicted)
            else:
                print("Invalid Input!")

            index = []

            if self.samplingtype == "mixed":
                for i in range(0, self.jump // 2):
                    index.append(values_lc[i][0])
                    x.append(X_train_copy[values_lc[i][0]])
                    y.append(y_train_copy[values_lc[i][0]])

                for i in range(0, self.jump // 2):
                    index.append(values_hc[i][0])
                    x.append(X_train_copy[values_hc[i][0]])
                    y.append(y_train_copy[values_hc[i][0]])


            else:
                for i in range(0, self.jump):
                    index.append(values[i][0])
                    x.append(X_train_copy[values[i][0]])
                    y.append(y_train_copy[values[i][0]])

            num += 1

            h = self.model.fit(self.datagen.flow(np.array(x), np.array(y), batch_size=self.batch_size),
                               validation_data=self.val_it,
                               epochs=self.epochs, verbose=1, workers=4,
                               callbacks=self.callbacks)

            batch_size_data.append(np.array(x).shape[0])
            epoch_data.append(num)
            history_data.append(h)

            eval_metrics = self.model.evaluate(self.X_test, self.y_test)
            print(eval_metrics)
            acuracy_data.append(100 * float(eval_metrics[1]))

            a = []
            for element in total_index:
                if element not in index:
                    a.append(element)

            X_train_copy = X_train_copy[a]
            y_train_copy = y_train_copy[a]

        self.model.save(
            os.getcwd() + f"/trained_models/{self.model_type}_{self.dataset}_{self.samplingtype}_{self.date}_completed")

        return history_data, epoch_data, batch_size_data, acuracy_data
