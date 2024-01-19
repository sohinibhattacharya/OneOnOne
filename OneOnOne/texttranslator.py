from pydub import AudioSegment
import speech_recognition as sr
import pyttsx3
import pytesseract
from googletrans import Translator
from io import BytesIO
from base64 import b64decode
from google.colab import output
from IPython.display import Javascript
import PyPDF2

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pkg_resources import resource_filename
from scipy.stats import randint

from torchvision import transforms

import fastapi
import kaleido
import python-multipart
import uvicorn

class TextTranslator:
    def __init__(self, file_bool=False,filepath="", translate_from_language="ben", translate_to_language="en", speak_bool=False):
        self.file_bool=file_bool
        self.filepath=filepath
        self.translate_from_language = translate_from_language
        self.translate_to_language = translate_to_language
        self.speak_bool = speak_bool

        if self.speak_bool:
            try:
                os.system("sudo apt install espeak")
                os.system("sudo apt install libespeak-dev")
                os.system("pip install pyaudio")
                os.system("sudo apt install python3-pyaudio")
                os.system("sudo apt install portaudio19 - dev")

            except:
                os.system("!sudo apt install espeak")
                os.system("!sudo apt install libespeak-dev")
                os.system("!pip install pyaudio")
                os.system("!sudo apt install python3-pyaudio")
                os.system("!sudo apt install portaudio19 - dev")

    def speak(self, command):
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

    def translate(self):

        if self.file_bool:
            file = open(os.getcwd()+f"/{self.filepath}","r")
            text=file.readlines()[0]
            print(file.readlines())
        else:
            text=input("What do you want to translate?:     ")

        translator = Translator()

        k = translator.translate(text, dest=self.translate_to_language)
        with open(f'{self.imgpath}_text_{self.translate_to_language}.txt', mode='w') as file:
            file.write(k.text)
        print(k.text)

        if self.speak_bool:
            self.speak(k.text)
