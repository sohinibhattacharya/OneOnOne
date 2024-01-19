
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pkg_resources import resource_filename
from scipy.stats import randint

from torchvision import transforms

# from tqdm import keras


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

