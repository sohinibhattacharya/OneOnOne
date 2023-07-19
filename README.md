
<h1 align="center">OneOnOne 🎈</h1>
<p>
  <a href="https://pypi.org/project/OneOnOne/0.6896/" target="_blank">
  </a>
</p>

<h3 align="center"><span style="color:#FFC0CB">Your one-stop destination to utilize image-classification models with just one line of code.</span></h3>

<h4 align="center">
A library meant to simplify your life by providing you with pre-trained models like ResNet50, EfficientNetVB6, VGG19, etc.
You can simply opt for training your own models from scratch by just tweaking a few values. If you want to try popular active-learning sampling methods on image classification, don't worry! This library has got you covered. Along with that for simple-bridging of NLP with Image Classification and utilization of basic use-cases of NLP frameworks, we have context-deciders, HTML parsers and simple chatbot object classes, to create an interface similar to Google Lens. You input an image or item that you are curious about, and you can ask one-on-one questions from the chatbot. This is made possible by using the tiny imagenet dataset.
</h4>

<h4 align="center">____________________________________________________________________________</h4>

<dl>
  <dt><span style="color:#FFC0CB">How to import the library?</span></dt>
    <dd>Just run the following command on your terminal or online coding interface.
</dd>
</dl>

```
pip install OneOnOne
```


<dl>
  <dt><span style="color:#FFC0CB">An example use case:</span>
    <dd>This will use the default values for the class object, as mentioned in the original code.
</dd>
</dl>

```python
from OneOnOne import Classification

classifier=Classification(validation_split=0.4,early_stopping_patience=20)
```

<dl>
  <dt><span style="color:#FFC0CB">or,</span>
</dl>

```python
from OneOnOne import Sampling

sampling=Sampling("entropy")
sampling.initial_training()
```

<dl>
  <dt><span style="color:#FFC0CB">or, download pretrained models with your desired specifications!</span>
</dl>

```python
from OneOnOne import PretrainedModel

pretrained=PretrainedModel(model_type="resnet50", dataset="cifar10", samplingtype="leastconfidence")
```

<h4 align="center">____________________________________________________________________________</h4>


<h4>More Examples on NLP use-cases:</h4>

<dl>
  <dt><span style="color:#FFC0CB">If you want your predicted class(es) of the input image to be used as the context for your chatbot, simply run, </span>
</dl>

```python
from OneOnOne import ContextDecider

decider=ContextDecider(dataset="tinyimagenet", model_type="efficientnetb6", samplingtype="none", threshold=0.4)
list_of_context_words=decider.decide_context()
```

<dl>
  <dt><span style="color:#FFC0CB">If you want your predicted class(es) of the input image to be used as the context for your chatbot, simply run, </span>
</dl>

```python
from OneOnOne import HTMLparser
from OneOnOne import QuestionAnswer

parser=HTMLparser(list_of_context_words)
text=parser.get_context_text()

qa=QuestionAnswer(text,"bert")
qa.ask()
```


<h4 align="center">____________________________________________________________________________</h4>


This library is being actively updated and new features are being added frequently. New datasets and pre-trained models will be updated soon. Any bugs detected will be fixed asap.

Feel free to share your feedback! I would really appreciate it! ❤️️

