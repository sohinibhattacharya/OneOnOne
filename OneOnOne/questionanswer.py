class QuestionAnswer:
    def __init__(self, context, chatbot="bert"):

        self.exit_commands = ("no", "n", "quit", "pause", "exit", "goodbye", "bye", "later", "stop")
        self.positive_commands = ("y", "yes", "yeah", "sure", "yup", "ya", "probably", "maybe")
        self.max_length = 4096

        self.context = context
        self.chatbot = chatbot.lower()

        if self.chatbot == "bert":
            self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        elif self.chatbot == "gpt2":
            self.model = GPT2Model.from_pretrained("gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        elif self.chatbot == "ernie":
            self.model = ErnieModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
            self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")

        elif self.chatbot == "roberta":
            self.model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
            self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

        elif self.chatbot == "vqa":
            self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

        else:
            print("invalid input")

    def question_answer(self, question):

        input_ids = self.tokenizer.encode(question, self.context)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        sep_idx = input_ids.index(self.tokenizer.sep_token_id)

        num_seg_a = sep_idx + 1

        num_seg_b = len(input_ids) - num_seg_a

        segment_ids = [0] * num_seg_a + [1] * num_seg_b

        assert len(segment_ids) == len(input_ids)

        output = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)

        if answer_end >= answer_start:
            answer = tokens[answer_start]
            for i in range(answer_start + 1, answer_end + 1):
                if tokens[i][0:2] == "##":
                    answer += tokens[i][2:]
                else:
                    answer += " " + tokens[i]

        if answer.startswith("[CLS]"):
            answer = "Sorry! Unable to find the answer to your question. Please ask another question."

        answer = "\nAnswer:\n{}".format(answer.capitalize())

        return answer

    def ask(self):

        while True:
            flag = True
            question = input("\nPlease enter your question: \n")

            if question.lower() in self.exit_commands:
                print("\nBye!")
                flag = False

            if not flag:
                break

            print(self.question_answer(question))





