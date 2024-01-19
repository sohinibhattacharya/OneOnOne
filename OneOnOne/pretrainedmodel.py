import pkgutil
import requests, zipfile
from io import BytesIO
import gdown
import zipfile
from zipfile import ZipFile
from tensorflow.keras.models import load_model

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
