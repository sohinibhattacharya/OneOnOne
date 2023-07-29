class ImageTranslator:
    def __init__(self, imgpath, translate_from_language="ben", translate_to_language="en", speak_bool=False):
        self.translate_from_language=translate_from_language
        self.translate_to_language=translate_to_language
        self.imgpath=imgpath
        self.speak_bool=speak_bool

    def speak(self, command):

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


        try:
            os.system("sudo apt install tesseract-ocr")
            os.system("apt install libtesseract-dev")
            os.system(f"apt install tesseract-ocr-{self.translate_from_language}")
            os.system(f"apt install tesseract-ocr-{self.translate_to_language}")
        except:
            os.system("!sudo apt install tesseract-ocr")
            os.system("!apt install libtesseract-dev")
            os.system(f"!apt install tesseract-ocr-{self.translate_from_language}")
            os.system(f"!apt install tesseract-ocr-{self.translate_to_language}")

        img = Image.open(os.getcwd()+"/"+self.imgpath)

        result = pytesseract.image_to_string(img,lang=self.translate_from_language)
        with open(f'{self.imgpath}_text_{self.translate_from_language}.txt', mode='w') as file:
            file.write(result)
            print(result)

        translator = Translator()

        k = translator.translate(result.replace("\n"," ")[:-5], dest=self.translate_to_language)
        with open(f'{self.imgpath}_text_{self.translate_to_language}.txt', mode='w') as file:
            file.write(k.text)
        print(k.text)

        if speak_bool:
            self.speak(k.text)