class ImageTranslator:
    def __init__(self, imgpath=imgpath, translate_from_language="english", translate_to_language="english", speak=False):
        self.translate_from_language=translate_from_language
        self.translate_to_language=translate_to_language
        self.imgpath=imgpath
        self.speak=self.speak
        self.date = datetime.datetime.now()

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
        except:
            os.system("!sudo apt install tesseract-ocr")
            os.system("!apt install libtesseract-dev")

        img = Image.open(os.getcwd()+self.imgpath)

        result = pytesseract.image_to_string(img,lang=self.translate_from_language)
        with open(f'{self.imgpath}_text_{self.translate_from_language}.txt', mode='w') as file:
            file.write(result)
            print(result)

        translator = Translator()

        k = translator.translate(result.replace("\n"," ")[:-5], dest=self.translate_to_language)
        print(k)

        if speak:
            self.speak(k)