from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
review = r"The room was kind of clean but had a VERY strong smell of dogs. Generally below average but ok for a overnight stay if you're not too fussy. Would consider staying again if the price was right. Breakfast was free and just about better than nothing."
tokenizedReview = sent_tokenize(review)

def tokenizeReview(review):
    return sent_tokenize(review)

def removeStopWordsFromReview(review):
    wordsReview = word_tokenize(review)
    newReview = ''
    for words in wordsReview:
        words = lemmatizer.lemmatize(words)
        if words not in stop_words:
            newReview = newReview + " " +  words

    return newReview



print removeStopWordsFromReview(review)