# <span style="color:#FFC0CB">OneOnOne</span>
## Your one-stop destination to utilize image-classification models with just one line of code.

A library meant to simplify your life by providing you with pre-trained models like ResNet50, EfficientNetVB6, VGG19, etc.
You can simply opt for training your own models from scratch by just tweaking a few values.

If you want to try popular active-learning sampling methods on image classification, don't worry! This library has got you covered.

Along with that for simple-bridging of NLP with Image Classification and utilization of basic use-cases of NLP frameworks, we have context-deciders, HTML parsers and simple chatbot object classes, to create an interface similar to Google Lens.  

You input an image or item that you are curious about and you can ask one-on-one questions from the chatbot. This is made possible by using the tiny imagenet dataset.

<dl>
  <dt><span style="color:#FFC0CB">How to import the library?</span></dt>
    <dd>Just run the following command on your terminal or online coding interface.
</dd>
</dl>

~~~~
pip install OneOnOne
~~~~

<dl>
  <dt><span style="color:#FFC0CB">An example use case:</span>
    <dd>This will use the default values for the class object, as mentioned in the original code.
</dd>
</dl>

~~~~
from OneOnOne import Classification
classifier=Classification(validation_split=0.4,early_stopping_patience=20)
~~~~

<dl>
  <dt><span style="color:#FFC0CB">or,</span>
</dl>

~~~~
from OneOnOne import Sampling
sampling=Sampling("entropy")
sampling.initial_training()
~~~~

<dl>
  <dt><span style="color:#FFC0CB">or, download pretrained models with your desired specifications!</span>
</dl>

~~~~
from OneOnOne import PretrainedModel
pretrained=PretrainedModel(model_type="resnet50", dataset="cifar10", samplingtype="leastconfidence")
~~~~


This library is being actively updated and new features are being added frequently.
New datasets and pre-trained models will be updated soon.

Feel free to share your feedback! I would really appreciate it!

