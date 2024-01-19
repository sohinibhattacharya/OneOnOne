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
# import python-multipart
import uvicorn

class PDFtoText:
    def __init__(self, pdfpath, speak_bool=False):
        self.pdfpath = imgpath
        self.speak_bool = speak_bool

    def convert(self, command):

        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

        if speak:
            try:
                os.system("sudo apt install espeak")
                os.system("sudo apt install libespeak-dev")
            except:
                os.system("!sudo apt install espeak")
                os.system("!sudo apt install libespeak-dev")

        path = open(os.getcwd() + f'/{self.pdfpath}', 'rb')

        pdfReader = PyPDF2.PdfFileReader(path)

        output = ""
        for i in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(i)
            output += pageObj.extractText()

        with open(f'{self.pdfpath}_text.txt', mode='w') as file:
            file.write(output)

        print(output)

        if self.speak_bool:
            self.speak(k.text)

        return output