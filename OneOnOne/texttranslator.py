class TextTranslator:
    def __init__(self, file_bool=False,filepath="", translate_from_language="ben", translate_to_language="en", speak_bool=False):
        self.file_bool=file_bool
        self.filepath=filepath
        self.translate_from_language = translate_from_language
        self.translate_to_language = translate_to_language
        self.speak_bool = speak_bool

        if self.file_bool:
            file = open(os.getcwd()+f"/{self.filepath}","r")
            text=file.readlines()[0]
            print(file.readlines())
        else:
            text=input("What do you want to translate?:     ")

        k = translator.translate(text, dest=self.translate_to_language)
        with open(f'{self.imgpath}_text_{self.translate_to_language}.txt', mode='w') as file:
            file.write(k.text)
        print(k.text)

        if speak_bool:
            self.speak(k.text)