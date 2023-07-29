
class PDFtoText:
    def __init__(self, pdfpath, speak_bool=False):
        self.pdfpath=imgpath
        self.speak_bool=speak_bool

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

        path = open(os.getcwd()+f'/{self.pdfpath}', 'rb')

        pdfReader = PyPDF2.PdfFileReader(path)

        output = ""
        for i in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(i)
            output += pageObj.extractText()

        with open(f'{self.pdfpath}_text.txt', mode='w') as file:
            file.write(output)

        print(output)

        if speak_bool:
            self.speak(k.text)

        return output