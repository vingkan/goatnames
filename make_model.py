import os 
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

dir_path = os.path.dirname(os.path.realpath(__file__))
glove_file = f"{dir_path}/embeddings/glove.6B/glove.6B.100d.txt"
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
model.save("models/glove.6B.100d.kv")
