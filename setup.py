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
setup(name="OneOnOne",version=VERSION,description=DESCRIPTION,author="Sohini Bhattacharya",author_email="mail.sohinibhattacharya@gmail.com",License="MIT",packages=find_packages(),keywords=["python","image-classification","active-learning-sampling","question-answering","pre-trained models","tiny-image-net","cifar10"],classifiers=CLASSIFIERS,install_requires=["wget","numpy","pandas","tensorflow","sklearn","datetime","os","zipfile","keras","math","torch","tensorflow_datasets","scipy","pickle","tqdm"])