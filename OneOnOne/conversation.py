class Conversation:
    def __init__(self):
        warnings.filterwarnings("ignore")

        self.conversational_pipeline = pipeline("conversational")

        try:
            os.system("apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg")
        except:
            os.system("!apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg")

        self.recognizer = sr.Recognizer()

        self.RECORD = """
                const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
                const b2text = blob => new Promise(resolve => {
                  const reader = new FileReader()
                  reader.onloadend = e => resolve(e.srcElement.result)
                  reader.readAsDataURL(blob)
                })
                var record = time => new Promise(async resolve => {
                  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
                  recorder = new MediaRecorder(stream)
                  chunks = []
                  recorder.ondataavailable = e => chunks.push(e.data)
                  recorder.start()
                  await sleep(time)
                  recorder.onstop = async ()=>{
                    blob = new Blob(chunks)
                    text = await b2text(blob)
                    resolve(text)
                  }
                  recorder.stop()
                })
                """
    def record(self,sec=3):
        display(Javascript(RECORD))
        sec += 1
        s = output.eval_js('record(%d)' % (sec*1000))
        b = b64decode(s.split(',')[1])
        return b

    def speak(self,command):

        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

    def answer(self,question):
        output = self.conversational_pipeline([question])
        out_list = str(output).split("\n")
        temp = out_list[2].split(">>")
        output = temp[1].replace("\n", "")
        return output

    def converse(self):

        while True:

            try:

                audio_source = sr.AudioData(self.record(), 16000, 2)

                question = self.recognize_google(audio_data=audio_source,language = 'en-IN')
                question = question.lower()

                print(f"{question}?")
                # self.speak(question)

                answer_output=self.answer(question)
                self.speak(answer_output)
                #
                # with sr.Microphone() as source2:
                #
                #     r.adjust_for_ambient_noise(source2,duration=5)
                #     print("Listening...")
                #
                #     audio2 = r.listen(source2,timeout=5,phrase_time_limit=5)
                #
                #     print("Recognizing...")
                #
                #     question = self.recognizer.recognize_google(audio2,language = 'en-IN')
                #     question = question.lower()
                #
                #     print("Did you say " + question)
                #     self.speak(question)
                #
                #     answer_output=self.answer(question)
                #     self.speak(answer_output)

            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

            except sr.UnknownValueError:
                print("Unknown Error Occured")



