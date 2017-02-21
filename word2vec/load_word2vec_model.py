import gensim

model = gensim.models.Word2Vec.load_word2vec_format('dblp.vector', binary=False)
model.most_similar("Acta")
