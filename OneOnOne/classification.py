
class Classification:
    def __init__(self, model_type="resnet50", batch_size=16, epochs=250, dataset="cifar10", validation_split=0.3, shuffle_bool=True, early_stopping_patience=10, lr_reducer_patience=10):

        self.model_type = model_type
        self.date = datetime.datetime.now()
        self.dataset=dataset
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


        elif self.dataset=="tinyimagenet":
            self.output_layer_classes = 200
            self.input_shape = (32, 32, 3)
            from classes import i2d
            self.download_dataset()

        else:
            print("Invalid Input!")

        self.token=input("Train/Load?  :   ")
        self.train_it, self.val_it = self.get_dataset()

        if self.token.lower() == 'train':
            print("training")

            self.model = self.define_compile_model()
            self.model.summary()

            history = self.model.fit(self.train_it, validation_data=self.val_it,
                                     epochs=self.epochs, verbose=1, workers=4,
                                     callbacks=self.callbacks)

            self.model.save(os.getcwd()+f"/fully_trained/{self.model_type}_{self.dataset}_{self.date}")

            history_token = input("Save/Discard?  :   ")

            if history_token.lower()=="save":
                keys,history_list=self.get_history_details(history)
                self.save_history_details(history_list,keys)
                print("saved")
            elif history_token.lower()=="save":
                print("discarded")
            else:
                print("Invalid Input!")

        elif self.token.lower() == 'load':
            print("loading")

            print(os.getcwd())

            self.model = load_model(os.getcwd()+"models_to_load/"+f'{self.model_type}_{self.dataset}_none')

        else:
            print("invalid input")

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
        print('Learning rate: ', lr)

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
            print("Invalid argument for model type")

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

        save_dir = os.path.join(os.getcwd(), f'saved_models_{self.model_type}_{self.dataset}_{self.date}')
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

        print("extracting")
        if not 'tiny-imagenet-200.zip' in os.listdir(os.getcwd()):
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            tiny_imgdataset = wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip', out=os.getcwd())

        for file in os.listdir(os.getcwd()):
            if file.endswith("tiny-imagenet-200.zip"):
                zip_file = ZipFile(file)
                zip_file.extractall()
            else:
                print("not found")

    def get_dataset(self):

        if self.dataset=="tinyimagenet":
            train_it = self.datagen.flow_from_directory('/home/sb355/testing/tiny-imagenet-200/train',
                                                        batch_size=self.batch_size, subset="training", shuffle=self.shuffle_bool)
            val_it = self.datagen.flow_from_directory('/home/sb355/testing/tiny-imagenet-200/train', batch_size=self.batch_size,
                                                      subset="validation", shuffle=self.shuffle_bool)
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            print(number_of_train_samples)
            print(number_of_val_samples)
        else:
            (train_it), (val_it) = tf.keras.datasets.cifar10.load_data()


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

    def context_decider_for_bert(self, val_it, prediction_list):
        pred = self.model.predict_generator(val_it, 1)

        predicted_list = []
        classes_prob_list = []
        prediction_index = []
        final_classes = []

        output = []

        labels = self.val_it.class_indices
        labels2 = dict((v, k) for k, v in labels.items())

        for i in range(0, pred.shape[0]):
            classes_prob_list = []
            for j in range(0, self.output_layer_classes):
                classes_prob_list.append([int(j), pred[i][j]])

            classes_prob_list.sort(key=lambda x: x[1], reverse=True)

            first_highest_predicted_classes = classes_prob_list[0][0]
            first_highest_predicted_class_confidence = classes_prob_list[0][1]

            second_highest_predicted_classes = classes_prob_list[1][0]
            second_highest_predicted_class_confidence = classes_prob_list[1][1]

            predicted_list.append([first_highest_predicted_classes, first_highest_predicted_class_confidence,
                                   second_highest_predicted_classes, second_highest_predicted_class_confidence])

        if (predicted_list[0][1] - predicted_list[0][3]) >= 0.2:
            prediction_index.append(predicted_list[0][0])
        elif (predicted_list[0][1] - predicted_list[0][3]) < 0.2:
            prediction_index.append(predicted_list[0][0])
            prediction_index.append(predicted_list[0][2])

        for i in range(0, len(prediction_index)):
            final_classes.append(i2d[labels2[prediction_index[i]]])

        for ele in final_classes:
            b = ele.split(',')
            output = output + b

        return output


    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.val_it, batch_size=self.batch_size, verbose=1)

