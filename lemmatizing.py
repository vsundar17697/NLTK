from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('better'))
print(lemmatizer.lemmatize('better' , pos="a"))
print(lemmatizer.lemmatize('was' ))
print(lemmatizer.lemmatize('run'))
print(lemmatizer.lemmatize('run' , pos="v"))