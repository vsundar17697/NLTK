from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()

# example_words = {"Python" , "Pythoner" , "Pythoning " , "Pyhtoned" , "Pythonly" , "Sex" , "Sexy"}

# for w in example_words:
#     print (w , ps.stem(w))


new_text = "It is a very important thing to be pythonly while we are pythoning with our sexy computer or watching others have sex"

words = word_tokenize(new_text)

for w in words:
    print(w , ps.stem(w))
