from setuptools import setup, find_packages

VERSION=0.0
DESCRIPTION="A package for pre-trained image classification and context-decider for question-answering chatbots."
LONG_DESCRIPTION=""
CLASSIFIERS=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]
setup(name="1on1",version='0.0',author="Sohini Bhattacharya",author_email="mail.sohinibhattacharya@gmail.com",License="MIT",packages=find_packages(),keywords=["python","image-classification","active-learning-sampling","question-answering","pre-trained models","tiny-image-net","cifar10"],classifiers=CLASSIFIERS)