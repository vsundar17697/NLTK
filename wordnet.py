from nltk.corpus import wordnet

syns = wordnet.synsets('program')

print syns[0] , syns[0].name()
print syns[0].lemmas()[0].name()

#definitions of words
print syns[1].definition()

#get examples
print syns[0].examples()

#all antonyms of good
synonymns = []
antonyms = []
for syn in wordnet.synsets("good"):
    for lemma in syn.lemmas():
        synonymns.append(lemma.name())
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())

print set(synonymns)
print set(antonyms)

#similarity between words (note synset and not synsets)
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2))