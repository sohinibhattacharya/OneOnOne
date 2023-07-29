class PretrainedModel:
    def __init__(self, model_type="resnet50", dataset="cifar10", samplingtype="none"):
        warnings.filterwarnings("ignore")

        self.model_type=model_type.lower()
        self.dataset=dataset.lower()
        self.samplingtype=samplingtype.lower()

        self.map={"resnet50_cifar10_none":"1YVG0lAnpBfM_MB7ctV1vHcslb6O-3Vbm","resnet50_cifar10_leastconfidence":"1fSJYo5VOppTkgWb-Fu2Sf_JL4s9JUmoK","resnet50_cifar10_none_initial_10k":"1NOicX1WgVzgRPWwWM_LzPyNvnu5st5Kd","resnet50_cifar10_mixedbayes":"","resnet50_cifar10_margin":"","efficientnetb6_cifar10_none":"16Wqfx7mcEhssZksF1yUAUEG6mkhMSeme","efficientnetb6_tinyimagenet_none":"1pGX8zB99ugqcvPohxykPC1JLM0Ld0L-D"}

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

    def get_model(self):

        self.model=load_model(self.path)
        self.model.summary()

        return self.model
