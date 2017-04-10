import nltk

l2 = 'Wei-Hsi Hung,Kuanchin Chen and Chieh-Pin Lin,Does the proactive personality mitigate the adverse effect of technostress on productivity in the mobile environment,Telematics and Informatics,2015,32(6),143-157'

words = nltk.word_tokenize('1 4 3 - 1 5 7')
print(words)
word_tag = nltk.pos_tag(words)
print(word_tag)