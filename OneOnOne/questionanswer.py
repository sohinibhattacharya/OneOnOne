from transformers import pipeline
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import GPT2Tokenizer, GPT2ForQuestionAnswering
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, RobertaForQuestionAnswering
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import AutoTokenizer, ErnieModel
from transformers import pipeline, Conversation

class QuestionAnswer:
    def __init__(self, context, chatbot="bert"):
        warnings.filterwarnings("ignore")

        self.exit_commands = ("no", "n", "quit", "pause", "exit", "goodbye", "bye", "later", "stop")
        self.positive_commands = ("y", "yes", "yeah", "sure", "yup", "ya", "probably", "maybe")
        self.context = context
        self.max_length = len(self.context)
        self.chatbot = chatbot.lower()

        if self.chatbot == "bert":
            self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        elif self.chatbot == "gpt2":
            self.model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        elif self.chatbot == "roberta":
            self.model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
            self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

        # elif self.chatbot == "ernie":
        #     self.model = ErnieModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
        #     self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")

        # elif self.chatbot == "vqa":
        #     self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
        #     self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

        else:
            print("invalid input")

    def question_answer(self, question):

        if self.chatbot=="bert":
            c = self.context[:512]

            input_ids = self.tokenizer.encode(question, c, add_special_tokens=True, truncation=True, max_length=len(context))

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # tokens = self.tokenizer.tokenize(self.context, max_length=self.max_length, truncation=True)

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

            if answer.startswith("[CLS]") or answer=="":
                answer = "Sorry! Unable to find the answer to your question. Please ask another question."

            answer = f"\nAnswer:\n{format(answer.capitalize())}"

        elif self.chatbot=="gpt2" or self.chatbot=="roberta":
            inputs = self.tokenizer(question, self.context, return_tensors="pt", truncation='longest_first', )

            input_ids = self.tokenizer.encode(question, self.context, max_length=self.max_length)

            with torch.no_grad():
                outputs = self.model(**inputs)

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            output_ids = self.tokenizer.decode(predict_answer_tokens)

            if output_ids == "" or output_ids.startswith("[CLS]") or output_ids== "<s>":
                output_ids = "Sorry! Unable to find the answer to your question. Please ask another question."

            answer = f"\nAnswer:\n {format(output_ids.capitalize())}"

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

