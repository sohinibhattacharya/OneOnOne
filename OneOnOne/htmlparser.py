
class HTMLparser:
    def __init__(self, words):
        self.words = words
    def clean_html(self,raw_html):
        clean_brackets = re.compile('<.*?>')
        cleantext = re.sub(clean_brackets, '', raw_html)
        cleantext = re.sub(' ,', '', cleantext)
        cleantext = re.sub('\n', '', cleantext)
        cleantext = cleantext.replace('\\', '')

        return cleantext

    def produce_text(self,word):
        link = "https://en.wikipedia.org/wiki/" + word

        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')

        p=soup.find_all('p')

        context=str(p)

        cleaned_text=self.cleanhtml(context)

        return cleaned_text


    def get_context(self):

        full_context = ""

        for word in self.words:

            text_for_one_word = self.produce_text(word)
            full_context = full_context + text_for_one_word[3:-1]

        return full_context
