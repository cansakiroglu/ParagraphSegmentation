import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from time import time
import os


#%%
nltk.download('punkt')
nltk.download("stopwords")

#%%
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#%%
with open("/home/mehmet/CS/NLP/wiki_00", "r") as f:
    text =f.readlines()
    text = "".join(text)


text = text.lower()

text_sent_tokenized = sent_tokenize(text)
#%%

#tagged_data = [TaggedDocument(word_tokenize(d), [i]) for i, d in enumerate(text_sent_tokenized[:-500000])]
tagged_data_small = [TaggedDocument(word_tokenize(d), [i]) for i, d in enumerate(text_sent_tokenized[:100000])]

#%%
model_small = Doc2Vec(tagged_data_small, vector_size=64, window=1, min_count=2, epochs=40 ,workers=11)
#%%
#t = time()
#model = Doc2Vec(tagged_data_small, vector_size=64, window=10, min_count=4, epochs=5)
#print(f"Required time: { time() -t }")

#%%
model_small.build_vocab(tagged_data_small)


#%%
print(f" ve kelimesi: {model_small.wv.get_vecattr('ve', 'count')} ")


#%%
model_small.train(tagged_data_small, total_examples=model_small.corpus_count, epochs=model_small.epochs)



#%%
vec1 = model_small.infer_vector(["cengiz", "han","ülkesini","başarılı", "bir", "şekilde", "savunmuştur"])
vec2 = model_small.infer_vector(["Bu","savaşlarda","büyük", "kahramanlıklar","göstermiştir"])
vec3 = model_small.infer_vector(["cengiz", "han","12", "çocuk", "sahibidir"])
#vec2 = model_small.infer_vector(["Is", "miss", "smith", "are", "with", "you", "today"])
#vec3 = model_small.infer_vector(["Today", "I", "did", "something", "for", "you" ])
#vec4 = model_small.infer_vector(["He", "worked","on", "a", "train" ])


#%%
import numpy as np
from numpy.linalg import  norm
def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))



sim1 = cos_similarity(vec1, vec2)
sim2 = cos_similarity(vec2, vec3)
sim3 = cos_similarity(vec1, vec3)


#%%
#Write results.
import tempfile

with open(tempfile.NamedTemporaryFile(prefix="doc2vec-model-small-results-",delete = False).name , "w+")  as resultTmp:
    similarities = f"Cosine similarities: v1-v2: {sim1}   v2-v3: {sim2}    v1-v3:{sim3}"
    hyperparams = f"Model params: Epochs:{model_small.epochs}"
    resultText = similarities + "\n" + hyperparams
    resultTmp.write(resultText)

#%%
import tempfile

with tempfile.NamedTemporaryFile(prefix="doc2vec-model-small-iter1",delete = False) as tmp:
    temporary_filepath = tmp.name
    model_small.save(temporary_filepath)



#%%

with open(tempfile.NamedTemporaryFile(prefix="sentence-tokenizer-results", delete=False).name, "w+") as resultTmp:
    resultText = ""

    for i in range(100):
        resultText += text_sent_tokenized[i] + "\n"

    resultTmp.write(resultText)


#%%
from keras2vec.keras2vec import Keras2Vec
from keras2vec.document import Document
import numpy as np