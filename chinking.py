#noun phrases descriptive group of words surrounding noun using regular expression ; mainly using modifiers part of regex


from nltk.corpus import state_union
import nltk
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)


            #nnp = proper noun nn = noun
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}
                                                }<VB.?|DT|IN|TO>+{"""
            chunkParse = nltk.RegexpParser(chunkGram)
            chunked = chunkParse.parse(tagged)

            # print tagged
            print chunked
            chunked.draw()
    except Exception as e:
        print e

process_content()