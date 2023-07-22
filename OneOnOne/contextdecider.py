class ContextDecider:
    def __int__(self, model_type="efficientnetb6", dataset="tinyimagenet", samplingtype="none", threshold=0.2):
        self.dataset = dataset.lower()
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.samplingtype=samplingtype
        self.train_it, self.val_it = self.get_dataset()

        self.pretrained = PretrainedModel(model_type=self.model_type, dataset=self.dataset,
                                          samplingtype=self.samplingtype)
        # self.pretrained.model
        self.pred = self.pretrained.model.predict_generator(self.val_it, 1)

        if self.dataset == "cifar10" or self.dataset == "mnist":
            self.output_layer_classes = 10
            self.input_shape = (224, 224, 3)

        elif self.dataset == "cifar100":
            self.output_layer_classes = 100
            self.input_shape = (224, 224, 3)

        elif self.dataset == "tinyimagenet":
            self.output_layer_classes = 200
            self.input_shape = (32, 32, 3)
            from classes import i2d
            self.download_dataset()

        else:
            print("Invalid Input!")


    def decide_context(self):

        predicted_list = []
        classes_prob_list = []
        prediction_index = []
        final_classes = []

        output = []

        labels = self.val_it.class_indices
        labels2 = dict((v, k) for k, v in labels.items())

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

        print("extracting")
        if not 'tiny-imagenet-200.zip' in os.listdir(os.getcwd()):
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            tiny_imgdataset = wget.download(url, out=os.getcwd())

        for file in os.listdir(os.getcwd()):
            if file.endswith("tiny-imagenet-200.zip"):
                zip_file = ZipFile(file)
                zip_file.extractall()

    def get_dataset(self):

        if self.dataset=="tinyimagenet":
            train_it = self.datagen.flow_from_directory(os.getcwd()+'tiny-imagenet-200/train',
                                                        batch_size=self.batch_size, subset="training", shuffle=self.shuffle_bool)
            val_it = self.datagen.flow_from_directory(os.getcwd()+'tiny-imagenet-200/train', batch_size=self.batch_size,
                                                      subset="validation", shuffle=self.shuffle_bool)
            train_filenames = train_it.filenames
            val_filenames = val_it.filenames
            number_of_val_samples = len(val_filenames)
            number_of_train_samples = len(train_filenames)
            # class_mode='categorical',
            print(number_of_train_samples)
            print(number_of_val_samples)
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
